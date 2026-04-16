import torch
import torch.nn.functional as F
import math
from kernels.vortex_fused import launch_vortex_fused, BLOCK_SIZE, HEAD_DIM, CHAOS_DEPTH

from kernels.vortex_function import VortexHelixFunction


def eager_vortex(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, chaos_depth=CHAOS_DEPTH):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    s = s.masked_fill(causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    a = p @ v.float()

    alpha, beta, phi = chaos_scalars[0], chaos_scalars[1], chaos_scalars[2]
    b = a
    for _ in range(chaos_depth):
        b = torch.tanh(b @ proj_weight.float()) + alpha * torch.sin(beta * b + phi)

    c = a
    for _ in range(3):
        c = c * 0.7 + a

    gate = torch.softmax(mixer_gate, dim=0)
    mixed = gate[0] * a + gate[1] * b + gate[2] * c
    return mixed + chaos_perturb * torch.sin(mixed)


def _build_dense_causal_csr(B, H, T):
    # Kernel convention: for Q block i, row_ptr[i+1]-row_ptr[i] entries where the
    # last entry is a placeholder for the diagonal block (handled specially);
    # the preceding entries are the off-diagonal lower K-block indices 0..i-1.
    nq = T // BLOCK_SIZE
    row_ptr_list = [0]
    col_idx_list = []
    for i in range(nq):
        for j in range(i):
            col_idx_list.append(j)
        col_idx_list.append(i)  # placeholder for the diagonal slot
        row_ptr_list.append(len(col_idx_list))
    row_ptr = torch.tensor(row_ptr_list, device='cuda', dtype=torch.int32).reshape(1, 1, -1).expand(B, H, -1).contiguous()
    col_idx = torch.tensor(col_idx_list, device='cuda', dtype=torch.int32).reshape(1, 1, -1).expand(B, H, -1).contiguous()
    seq_lens = torch.tensor([T] * B, device='cuda', dtype=torch.int32)
    return row_ptr, col_idx, seq_lens


def test_kernel():
    B, H, T = 2, 4, 256
    print(f"CHAOS_DEPTH={CHAOS_DEPTH}")

    torch.manual_seed(0)
    q = torch.randn(B, H, T, HEAD_DIM, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, HEAD_DIM, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, HEAD_DIM, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    proj_weight = (torch.randn(HEAD_DIM, HEAD_DIM, device='cuda', dtype=torch.bfloat16) * 0.1).detach().requires_grad_(True)
    chaos_scalars = (torch.randn(3, device='cuda', dtype=torch.float32) * 0.3).detach().requires_grad_(True)
    mixer_gate = torch.randn(3, device='cuda', dtype=torch.float32, requires_grad=True)
    chaos_perturb = (torch.randn(1, device='cuda', dtype=torch.float32) * 0.1).detach().requires_grad_(True)

    row_ptr, col_idx, seq_lens = _build_dense_causal_csr(B, H, T)

    out = VortexHelixFunction.apply(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, row_ptr, col_idx, seq_lens)
    do = torch.randn_like(out)
    out.backward(do)
    print("Triton forward+backward ran.")
    print("dq:", q.grad.shape, "dk:", k.grad.shape, "dv:", v.grad.shape)
    print("dproj_weight:", proj_weight.grad.shape, "dchaos_scalars:", chaos_scalars.grad.shape)
    print("dmixer_gate:", mixer_gate.grad.shape, "dchaos_perturb:", chaos_perturb.grad.shape)

    grads = {
        'q': q.grad.detach().clone(), 'k': k.grad.detach().clone(), 'v': v.grad.detach().clone(),
        'proj_weight': proj_weight.grad.detach().clone(),
        'chaos_scalars': chaos_scalars.grad.detach().clone(),
        'mixer_gate': mixer_gate.grad.detach().clone(),
        'chaos_perturb': chaos_perturb.grad.detach().clone(),
    }
    out_ref_triton = out.detach().clone()

    for p in (q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb):
        p.grad = None

    out_ref = eager_vortex(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb)
    out_ref.backward(do.float())

    print("\n=== Forward comparison (kernel vs eager) ===")
    _cmp("out", out_ref_triton.float(), out_ref)

    print("\n=== Backward comparison (kernel vs eager) ===")
    _cmp("dq", grads['q'].float(), q.grad.float())
    _cmp("dk", grads['k'].float(), k.grad.float())
    _cmp("dv", grads['v'].float(), v.grad.float())
    _cmp_opt("dproj_weight", grads['proj_weight'], proj_weight.grad)
    _cmp_opt("dchaos_scalars", grads['chaos_scalars'], chaos_scalars.grad)
    _cmp_opt("dmixer_gate", grads['mixer_gate'], mixer_gate.grad)
    _cmp_opt("dchaos_perturb", grads['chaos_perturb'], chaos_perturb.grad)


def _cmp(name, a, b):
    diff = (a - b).abs()
    max_a = a.abs().max().item()
    max_b = b.abs().max().item()
    rel = diff.max().item() / max(max_b, 1e-6)
    print(f"{name:18s} max_abs_diff={diff.max().item():.4e} mean={diff.mean().item():.4e} rel={rel:.4e} ref_max={max_b:.3e}")


def _cmp_opt(name, a, b):
    if a is None and b is None:
        print(f"{name:18s} both grads are None (no gradient path for this depth)")
        return
    if a is None or b is None:
        kernel_state = "None" if a is None else f"nonzero_max={a.abs().max().item():.3e}"
        eager_state = "None" if b is None else f"nonzero_max={b.abs().max().item():.3e}"
        print(f"{name:18s} MISMATCH kernel={kernel_state} eager={eager_state}")
        return
    _cmp(name, a.float(), b.float())


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_kernel()
    else:
        print("CUDA not available")
