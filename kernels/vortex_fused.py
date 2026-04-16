import math
import os
import torch
import triton
import triton.language as tl

BLOCK_SIZE = int(os.environ.get("VORTEX_BLOCK_SIZE", "64"))
HEAD_DIM = 128
CHAOS_DEPTH = int(os.environ.get("VORTEX_CHAOS_DEPTH", "5"))
CHAOS_STORE = int(os.environ.get("VORTEX_CHAOS_STORE", "0"))
_FWD_NUM_WARPS = int(os.environ.get("VORTEX_FWD_NUM_WARPS", "4"))
_FWD_NUM_STAGES = int(os.environ.get("VORTEX_FWD_NUM_STAGES", "1"))

@triton.jit
def _vortex_helix_fwd_kernel(
    Q, K, V,
    Proj_Weight,      # (D, D) for Stream B projection
    Chaos_Scalars,    # (3,) -> alpha, beta, phi
    Mixer_Gate,       # (3,)
    Chaos_Perturb,    # (1,)
    O, LSE,
    RowPtr, ColIdx, SeqLens,
    ChaosStore_B,     # [BH, CHAOS_DEPTH, T_MAX, D] fp32, or 1-elem placeholder
    ChaosStore_T,     # [BH, CHAOS_DEPTH, T_MAX, D] fp32, or 1-elem placeholder
    AStore,           # [BH, T_MAX, D] fp32 cache of A_i for bwd recompute skip
    stride_cih,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    CHAOS_DEPTH: tl.constexpr,
    STORE_CHAOS: tl.constexpr,
    STORE_A: tl.constexpr,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_lh: tl.constexpr = T_MAX
    stride_rph: tl.constexpr = (T_MAX // BS) + 1
    NUM_Q_BLOCKS: tl.constexpr = T_MAX // BS
    
    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    seq_len = tl.load(SeqLens + bh_id // NUM_HEADS)
    q_start = q_block_id * BS

    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)

    if q_start >= seq_len:
        O_ptr_e = O + bh_id * stride_h
        tl.store(O_ptr_e + offs_tok[:, None] * stride_t + offs_d[None, :], tl.zeros([BS, D], dtype=tl.bfloat16))
        LSE_ptr_e = LSE + bh_id * stride_lh
        tl.store(LSE_ptr_e + offs_tok, tl.full([BS], float("-inf"), dtype=tl.float32))
        return

    q_mask = offs_tok < seq_len
    Q_ptr = Q + bh_id * stride_h
    q_bf16 = tl.load(Q_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    m_i = tl.full([BS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BS], dtype=tl.float32)
    acc = tl.zeros([BS, D], dtype=tl.float32)

    rp_base = RowPtr + bh_id * stride_rph + q_block_id
    ci_lo = tl.load(rp_base)
    ci_hi = tl.load(rp_base + 1)

    K_ptr = K + bh_id * stride_h
    V_ptr = V + bh_id * stride_h
    CI_ptr = ColIdx + bh_id * stride_cih

    offs_d_arange = tl.arange(0, BS)
    LOG2E: tl.constexpr = 1.4426950408889634
    SCALE_2: tl.constexpr = SCALE * LOG2E
    boundary = (q_start + BS) > seq_len

    # Diagonal block
    if ci_hi > ci_lo:
        k_start_d = q_block_id * BS
        offs_k_d = k_start_d + offs_d_arange
        
        if boundary:
            k_mask = offs_k_d < seq_len
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE_2
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            s_d = tl.where(causal & k_mask[None, :], s_d, float("-inf"))
        else:
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE_2
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            s_d = tl.where(causal, s_d, float("-inf"))

        m_i = tl.max(s_d, axis=1)
        p_d = tl.exp2(s_d - m_i[:, None])
        l_i = tl.sum(p_d, axis=1)
        p_hi = p_d.to(tl.bfloat16)
        p_lo = ((p_d - p_hi.to(tl.float32)) * 128.0).to(tl.bfloat16)
        acc = tl.dot(p_hi, v_bf16_d, out_dtype=tl.float32)
        acc_lo = tl.dot(p_lo, v_bf16_d, out_dtype=tl.float32)
        acc = acc + acc_lo * (1.0 / 128.0)

    for ci in range(ci_lo, ci_hi - 1):
        k_block_id = tl.load(CI_ptr + ci)
        k_start = k_block_id * BS
        offs_k = k_start + offs_d_arange
        k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
        v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
        s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE_2
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v_bf16, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new

    LN2: tl.constexpr = 0.6931471805599453
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    a = tl.where(q_mask[:, None], acc / l_safe[:, None], 0.0) # Stream A
    lse_out = tl.where(q_mask, (m_i + tl.log2(l_safe)) * LN2, float("-inf"))

    if STORE_A:
        a_store_offs = bh_id * T_MAX * D + offs_tok[:, None] * D + offs_d[None, :]
        tl.store(AStore + a_store_offs, a)

    # VORTEX FUSION START
    # Load Projection Weight for Stream B (D x D)
    W_offs_x = tl.arange(0, D)[:, None]
    W_offs_y = tl.arange(0, D)[None, :]
    W = tl.load(Proj_Weight + W_offs_x * D + W_offs_y)
    
    # Stream B: Chaotic loop
    b_fp32 = a

    # Pre-load projection weight (assume transposed for tl.dot, or handled correctly)
    W_ptr = Proj_Weight + offs_d[:, None] * D + offs_d[None, :]
    W = tl.load(W_ptr).to(tl.bfloat16)

    alpha_val = tl.load(Chaos_Scalars + 0)
    beta_val  = tl.load(Chaos_Scalars + 1)
    phi_val   = tl.load(Chaos_Scalars + 2)

    # Fixed-point iterations (depth = constexpr CHAOS_DEPTH). When STORE_CHAOS
    # is set, also dump B_input and T for each iter so the bwd can skip the
    # forward matmul recompute.
    stride_cs_bh: tl.constexpr = CHAOS_DEPTH * T_MAX * D
    stride_cs_i: tl.constexpr = T_MAX * D
    for i_step in range(CHAOS_DEPTH):
        if STORE_CHAOS:
            store_offs = (bh_id * stride_cs_bh + i_step * stride_cs_i
                          + offs_tok[:, None] * D + offs_d[None, :])
            tl.store(ChaosStore_B + store_offs, b_fp32)
        b_bf16 = b_fp32.to(tl.bfloat16)
        b_proj = tl.dot(b_bf16, W, out_dtype=tl.float32)
        # Using sigmoid approximation for tanh to ensure compatibility: tanh(x) = 2*sigmoid(2x) - 1
        e_2x = tl.exp(2.0 * b_proj)
        b_tanh = (e_2x - 1.0) / (e_2x + 1.0)
        if STORE_CHAOS:
            store_offs = (bh_id * stride_cs_bh + i_step * stride_cs_i
                          + offs_tok[:, None] * D + offs_d[None, :])
            tl.store(ChaosStore_T + store_offs, b_tanh)
        b_sin  = tl.sin(beta_val * b_fp32 + phi_val)
        b_fp32 = b_tanh + alpha_val * b_sin

    # Stream C: Fractal Echo
    c_fp32 = a
    for _ in range(3):
        c_fp32 = c_fp32 * 0.7 + a

    # Helix Mixer
    mixer_0 = tl.load(Mixer_Gate + 0)
    mixer_1 = tl.load(Mixer_Gate + 1)
    mixer_2 = tl.load(Mixer_Gate + 2)
    # Simple softmax for gate
    max_gate = tl.maximum(tl.maximum(mixer_0, mixer_1), mixer_2)
    exp_0 = tl.exp(mixer_0 - max_gate)
    exp_1 = tl.exp(mixer_1 - max_gate)
    exp_2 = tl.exp(mixer_2 - max_gate)
    sum_exp = exp_0 + exp_1 + exp_2
    gate_0 = exp_0 / sum_exp
    gate_1 = exp_1 / sum_exp
    gate_2 = exp_2 / sum_exp

    mixed = gate_0 * a + gate_1 * b_fp32 + gate_2 * c_fp32

    perturb = tl.load(Chaos_Perturb + 0)
    out_final = mixed + perturb * tl.sin(mixed)

    O_ptr = O + bh_id * stride_h
    tl.store(O_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], out_final.to(tl.bfloat16))
    LSE_ptr = LSE + bh_id * stride_lh
    tl.store(LSE_ptr + offs_tok, lse_out)

def launch_vortex_fused(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, row_ptr, col_idx, seq_lens):
    # Helper wrapper to launch the kernel
    qshape = q.shape
    batch_size = qshape[0]
    num_heads = qshape[1]
    t_max = qshape[2]
    batch_heads = batch_size * num_heads
    num_q_blocks = t_max // BLOCK_SIZE

    o = torch.empty_like(q)
    lse_3d = torch.empty((batch_size, num_heads, t_max), device=q.device, dtype=torch.float32)
    lse_2d = lse_3d.view(batch_heads, t_max)

    if CHAOS_STORE and CHAOS_DEPTH > 0:
        chaos_store_B = torch.empty((batch_heads, CHAOS_DEPTH, t_max, HEAD_DIM),
                                    device=q.device, dtype=torch.float32)
        chaos_store_T = torch.empty((batch_heads, CHAOS_DEPTH, t_max, HEAD_DIM),
                                    device=q.device, dtype=torch.float32)
    else:
        # 1-elem placeholders; kernel never dereferences them when STORE_CHAOS=0.
        chaos_store_B = torch.empty(1, device=q.device, dtype=torch.float32)
        chaos_store_T = torch.empty(1, device=q.device, dtype=torch.float32)

    # A_i scratch: zero-init so masked/OOB rows read back as 0 in bwd.
    a_store = torch.zeros((batch_heads, t_max, HEAD_DIM),
                          device=q.device, dtype=torch.float32)

    grid = (num_q_blocks * batch_heads,)

    _vortex_helix_fwd_kernel[grid](
        q, k, v,
        proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
        o, lse_2d,
        row_ptr, col_idx, seq_lens,
        chaos_store_B, chaos_store_T,
        a_store,
        col_idx.shape[2] if col_idx.ndim == 3 else col_idx.shape[1],
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        SCALE=1.0 / math.sqrt(HEAD_DIM),
        BS=BLOCK_SIZE,
        D=HEAD_DIM,
        CHAOS_DEPTH=CHAOS_DEPTH,
        STORE_CHAOS=CHAOS_STORE,
        STORE_A=1,
        num_stages=_FWD_NUM_STAGES,
        num_warps=_FWD_NUM_WARPS,
    )
    return o, lse_3d, chaos_store_B, chaos_store_T, a_store
