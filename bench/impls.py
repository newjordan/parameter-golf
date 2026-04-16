"""Attention implementations under test.

Each impl exposes a `build(B,H,T,D,device,dtype)` factory that returns
`(leaves, fwd_fn, fwd_args)`:
  - leaves: leaf tensors with requires_grad=True (for bwd)
  - fwd_fn: callable(*fwd_args) -> output
  - fwd_args: tuple passed to fwd_fn

The fused vortex impl also accepts optional chaos config for ablations.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _mkqkv(B, H, T, D, device, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _eager_vortex_body(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, chaos_iters: int):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    s = s.masked_fill(causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    a = p @ v.float()
    alpha, beta, phi = chaos_scalars[0], chaos_scalars[1], chaos_scalars[2]
    b = a
    for _ in range(chaos_iters):
        b = torch.tanh(b @ proj_weight.float()) + alpha * torch.sin(beta * b + phi)
    c = a
    for _ in range(3):
        c = c * 0.7 + a
    gate = torch.softmax(mixer_gate, dim=0)
    mixed = gate[0] * a + gate[1] * b + gate[2] * c
    return (mixed + chaos_perturb * torch.sin(mixed)).to(q.dtype)


def build_eager(B, H, T, D, device, dtype, chaos_iters=5):
    q, k, v = _mkqkv(B, H, T, D, device, dtype)
    pw = (torch.randn(D, D, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    cs = (torch.randn(3, device=device, dtype=torch.float32) * 0.3).detach().requires_grad_(True)
    mg = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    cp = (torch.randn(1, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    leaves = [q, k, v, pw, cs, mg, cp]
    def fwd(q, k, v, pw, cs, mg, cp):
        return _eager_vortex_body(q, k, v, pw, cs, mg, cp, chaos_iters)
    return leaves, fwd, (q, k, v, pw, cs, mg, cp)


def build_sdpa_attn_only(B, H, T, D, device, dtype, backend: SDPBackend):
    q, k, v = _mkqkv(B, H, T, D, device, dtype)
    def fwd(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return [q, k, v], fwd, (q, k, v)


def build_sdpa_vortex(B, H, T, D, device, dtype, backend: SDPBackend, chaos_iters=5):
    q, k, v = _mkqkv(B, H, T, D, device, dtype)
    pw = (torch.randn(D, D, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    cs = (torch.randn(3, device=device, dtype=torch.float32) * 0.3).detach().requires_grad_(True)
    mg = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    cp = (torch.randn(1, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    leaves = [q, k, v, pw, cs, mg, cp]
    def fwd(q, k, v, pw, cs, mg, cp):
        with sdpa_kernel(backend):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).float()
        alpha, beta, phi = cs[0], cs[1], cs[2]
        b = a
        for _ in range(chaos_iters):
            b = torch.tanh(b @ pw.float()) + alpha * torch.sin(beta * b + phi)
        c = a
        for _ in range(3):
            c = c * 0.7 + a
        gate = torch.softmax(mg, dim=0)
        mixed = gate[0] * a + gate[1] * b + gate[2] * c
        return (mixed + cp * torch.sin(mixed)).to(q.dtype)
    return leaves, fwd, (q, k, v, pw, cs, mg, cp)


def build_vortex_fused(B, H, T, D, device, dtype):
    from kernels.vortex_fused import BLOCK_SIZE, HEAD_DIM
    from kernels.vortex_function import VortexHelixFunction
    assert D == HEAD_DIM, f"kernel requires HEAD_DIM={HEAD_DIM}, got D={D}"
    assert T % BLOCK_SIZE == 0, f"T={T} must be multiple of {BLOCK_SIZE}"
    q, k, v = _mkqkv(B, H, T, D, device, dtype)
    pw = (torch.randn(D, D, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    cs = (torch.randn(3, device=device, dtype=torch.float32) * 0.3).detach().requires_grad_(True)
    mg = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    cp = (torch.randn(1, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
    leaves = [q, k, v, pw, cs, mg, cp]

    # Dense causal CSR
    nq = T // BLOCK_SIZE
    row_ptr_list = [0]
    col_idx_list: list[int] = []
    for i in range(nq):
        for j in range(i):
            col_idx_list.append(j)
        col_idx_list.append(i)
        row_ptr_list.append(len(col_idx_list))
    row_ptr = torch.tensor(row_ptr_list, device=device, dtype=torch.int32).reshape(1, 1, -1).expand(B, H, -1).contiguous()
    col_idx = torch.tensor(col_idx_list, device=device, dtype=torch.int32).reshape(1, 1, -1).expand(B, H, -1).contiguous()
    seq_lens = torch.tensor([T] * B, device=device, dtype=torch.int32)

    def fwd(q, k, v, pw, cs, mg, cp):
        return VortexHelixFunction.apply(q, k, v, pw, cs, mg, cp, row_ptr, col_idx, seq_lens)
    return leaves, fwd, (q, k, v, pw, cs, mg, cp)
