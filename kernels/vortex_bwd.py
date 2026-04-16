import math
import os
import torch
import triton
import triton.language as tl
from kernels.vortex_fused import BLOCK_SIZE, HEAD_DIM, CHAOS_DEPTH, CHAOS_STORE

_BWD_CHAOS_NUM_WARPS = int(os.environ.get("VORTEX_BWD_CHAOS_NUM_WARPS", "4"))
_BWD_CHAOS_NUM_STAGES = int(os.environ.get("VORTEX_BWD_CHAOS_NUM_STAGES", "1"))
_BWD_ATTN_NUM_WARPS = int(os.environ.get("VORTEX_BWD_ATTN_NUM_WARPS", "4"))
_BWD_ATTN_NUM_STAGES = int(os.environ.get("VORTEX_BWD_ATTN_NUM_STAGES", "1"))
_BWD_ATTN_DKV_NUM_WARPS = int(os.environ.get("VORTEX_BWD_ATTN_DKV_NUM_WARPS", str(_BWD_ATTN_NUM_WARPS)))
_BWD_ATTN_DKV_NUM_STAGES = int(os.environ.get("VORTEX_BWD_ATTN_DKV_NUM_STAGES", "1"))
_BWD_PROFILE = int(os.environ.get("VORTEX_BWD_PROFILE", "0"))


def build_attn_bwd_csrT(row_ptr, col_idx, num_blocks):
    """Build a transposed CSR of the OFF-diagonal sparsity pattern.

    The fwd convention places the diagonal K-block as the final entry of each
    Q-row's col_idx slice. This helper enumerates, per (B,H), the Q-blocks
    that reference each K-block as an off-diagonal attendee, used by the
    K-parallel dK/dV kernel to accumulate without atomics.
    """
    import numpy as np
    rp = row_ptr.detach().cpu().numpy()
    ci = col_idx.detach().cpu().numpy()
    B, H = rp.shape[0], rp.shape[1]
    rp_flat = rp.reshape(B * H, -1)
    ci_flat = ci.reshape(B * H, -1)
    rpt_rows = []
    cit_rows = []
    for bh in range(B * H):
        bucket = [[] for _ in range(num_blocks)]
        rp_row = rp_flat[bh]
        ci_row = ci_flat[bh]
        for i in range(num_blocks):
            lo = int(rp_row[i]); hi = int(rp_row[i + 1])
            for p in range(lo, max(lo, hi - 1)):
                bucket[int(ci_row[p])].append(i)
        offs = [0]
        vals = []
        for k in range(num_blocks):
            vals.extend(bucket[k])
            offs.append(len(vals))
        rpt_rows.append(offs)
        cit_rows.append(vals)
    max_nnz = max(1, max(len(v) for v in cit_rows))
    cit_pad = np.zeros((B * H, max_nnz), dtype=np.int32)
    for bh, v in enumerate(cit_rows):
        if v:
            cit_pad[bh, :len(v)] = v
    rpt_np = np.asarray(rpt_rows, dtype=np.int32)
    rp_T = torch.from_numpy(rpt_np).to(row_ptr.device).reshape(B, H, num_blocks + 1).contiguous()
    ci_T = torch.from_numpy(cit_pad).to(row_ptr.device).reshape(B, H, max_nnz).contiguous()
    return rp_T, ci_T


@triton.jit
def _vortex_chaos_bwd_kernel(
    Q, K, V,
    Proj_Weight, Chaos_Scalars, Mixer_Gate, Chaos_Perturb,
    dO,
    dA_total_out,
    Di_out,
    dW_workspace, dScalars_workspace, dMixer_workspace, dPerturb_workspace,
    RowPtr, ColIdx, SeqLens,
    ChaosStore_B, ChaosStore_T,
    AStore,
    stride_cih,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    CHAOS_DEPTH: tl.constexpr,
    LOAD_CHAOS: tl.constexpr,
    LOAD_A: tl.constexpr,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_rph: tl.constexpr = (T_MAX // BS) + 1
    stride_lh: tl.constexpr = T_MAX
    NUM_Q_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    seq_len = tl.load(SeqLens + bh_id // NUM_HEADS)
    q_start = q_block_id * BS

    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)

    if q_start >= seq_len:
        dA_ptr_e = dA_total_out + bh_id * stride_h
        tl.store(dA_ptr_e + offs_tok[:, None] * stride_t + offs_d[None, :], tl.zeros([BS, D], dtype=tl.bfloat16))
        tl.store(Di_out + bh_id * stride_lh + offs_tok, tl.zeros([BS], dtype=tl.float32))
        return

    q_mask = offs_tok < seq_len
    LOG2E: tl.constexpr = 1.4426950408889634
    SCALE_2: tl.constexpr = SCALE * LOG2E

    if LOAD_A:
        # Load A_i cached by the fwd, skipping the entire sparsity recompute.
        a_offs = bh_id * stride_h + offs_tok[:, None] * stride_t + offs_d[None, :]
        A_i = tl.load(AStore + a_offs)
        A_i = tl.where(q_mask[:, None], A_i, 0.0)
    else:
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

        offs_k_tile = tl.arange(0, BS)
        boundary = (q_start + BS) > seq_len

        if ci_hi > ci_lo:
            k_start_d = q_block_id * BS
            offs_k_d = k_start_d + offs_k_tile
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
            acc = tl.dot(p_d.to(tl.bfloat16), v_bf16_d, out_dtype=tl.float32)

        for ci in range(ci_lo, ci_hi - 1):
            k_block_id = tl.load(CI_ptr + ci)
            k_start = k_block_id * BS
            offs_k = k_start + offs_k_tile
            k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE_2
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha_scale = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha_scale + tl.sum(p, axis=1)
            acc = tl.dot(p.to(tl.bfloat16), v_bf16, acc=acc * alpha_scale[:, None], out_dtype=tl.float32)
            m_i = m_new

        l_safe = tl.where(l_i > 0, l_i, 1.0)
        A_i = tl.where(q_mask[:, None], acc / l_safe[:, None], 0.0)

    W_ptr_base = Proj_Weight + offs_d[:, None] * D + offs_d[None, :]
    W_bf16 = tl.load(W_ptr_base).to(tl.bfloat16)
    WT_bf16 = tl.trans(W_bf16)

    alpha_val = tl.load(Chaos_Scalars + 0)
    beta_val = tl.load(Chaos_Scalars + 1)
    phi_val = tl.load(Chaos_Scalars + 2)

    mixer_0 = tl.load(Mixer_Gate + 0)
    mixer_1 = tl.load(Mixer_Gate + 1)
    mixer_2 = tl.load(Mixer_Gate + 2)
    max_gate = tl.maximum(tl.maximum(mixer_0, mixer_1), mixer_2)
    exp_0 = tl.exp(mixer_0 - max_gate)
    exp_1 = tl.exp(mixer_1 - max_gate)
    exp_2 = tl.exp(mixer_2 - max_gate)
    sum_exp = exp_0 + exp_1 + exp_2
    gate_0 = exp_0 / sum_exp
    gate_1 = exp_1 / sum_exp
    gate_2 = exp_2 / sum_exp

    perturb = tl.load(Chaos_Perturb + 0)

    # Forward recompute of the chaotic stream with constexpr-gated iterations.
    # Triton evaluates `if CHAOS_DEPTH >= N:` at compile time and DCEs the
    # untaken branch, so only the needed iterations emit IR. All B_i/T_i/S_i
    # are pre-initialised to A_i so downstream references never hit NameError.
    # When LOAD_CHAOS is set, B_{i-1} and T_i are loaded from the forward
    # scratch buffers instead of being recomputed (skips the per-iter matmul).
    stride_cs_bh: tl.constexpr = CHAOS_DEPTH * T_MAX * D
    stride_cs_i: tl.constexpr = T_MAX * D
    B0 = A_i
    B0_bf16 = B0.to(tl.bfloat16)
    B1 = A_i; T1 = A_i; S1 = A_i; B1_bf16 = B0_bf16
    B2 = A_i; T2 = A_i; S2 = A_i; B2_bf16 = B0_bf16
    B3 = A_i; T3 = A_i; S3 = A_i; B3_bf16 = B0_bf16
    B4 = A_i; T4 = A_i; S4 = A_i; B4_bf16 = B0_bf16
    B5 = A_i; T5 = A_i; S5 = A_i; B5_bf16 = B0_bf16
    B6 = A_i; T6 = A_i; S6 = A_i; B6_bf16 = B0_bf16
    B7 = A_i; T7 = A_i; S7 = A_i

    if CHAOS_DEPTH >= 1:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 0 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            T1 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_1 = tl.exp(2.0 * tl.dot(B0_bf16, W_bf16, out_dtype=tl.float32))
            T1 = (e_2x_1 - 1.0) / (e_2x_1 + 1.0)
        S1 = tl.sin(beta_val * B0 + phi_val)
        B1 = T1 + alpha_val * S1
        B1_bf16 = B1.to(tl.bfloat16)
    if CHAOS_DEPTH >= 2:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 1 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B1 = tl.load(ChaosStore_B + cs_offs)
            B1_bf16 = B1.to(tl.bfloat16)
            T2 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_2 = tl.exp(2.0 * tl.dot(B1_bf16, W_bf16, out_dtype=tl.float32))
            T2 = (e_2x_2 - 1.0) / (e_2x_2 + 1.0)
        S2 = tl.sin(beta_val * B1 + phi_val)
        B2 = T2 + alpha_val * S2
        B2_bf16 = B2.to(tl.bfloat16)
    if CHAOS_DEPTH >= 3:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 2 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B2 = tl.load(ChaosStore_B + cs_offs)
            B2_bf16 = B2.to(tl.bfloat16)
            T3 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_3 = tl.exp(2.0 * tl.dot(B2_bf16, W_bf16, out_dtype=tl.float32))
            T3 = (e_2x_3 - 1.0) / (e_2x_3 + 1.0)
        S3 = tl.sin(beta_val * B2 + phi_val)
        B3 = T3 + alpha_val * S3
        B3_bf16 = B3.to(tl.bfloat16)
    if CHAOS_DEPTH >= 4:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 3 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B3 = tl.load(ChaosStore_B + cs_offs)
            B3_bf16 = B3.to(tl.bfloat16)
            T4 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_4 = tl.exp(2.0 * tl.dot(B3_bf16, W_bf16, out_dtype=tl.float32))
            T4 = (e_2x_4 - 1.0) / (e_2x_4 + 1.0)
        S4 = tl.sin(beta_val * B3 + phi_val)
        B4 = T4 + alpha_val * S4
        B4_bf16 = B4.to(tl.bfloat16)
    if CHAOS_DEPTH >= 5:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 4 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B4 = tl.load(ChaosStore_B + cs_offs)
            B4_bf16 = B4.to(tl.bfloat16)
            T5 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_5 = tl.exp(2.0 * tl.dot(B4_bf16, W_bf16, out_dtype=tl.float32))
            T5 = (e_2x_5 - 1.0) / (e_2x_5 + 1.0)
        S5 = tl.sin(beta_val * B4 + phi_val)
        B5 = T5 + alpha_val * S5
        B5_bf16 = B5.to(tl.bfloat16)
    if CHAOS_DEPTH >= 6:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 5 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B5 = tl.load(ChaosStore_B + cs_offs)
            B5_bf16 = B5.to(tl.bfloat16)
            T6 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_6 = tl.exp(2.0 * tl.dot(B5_bf16, W_bf16, out_dtype=tl.float32))
            T6 = (e_2x_6 - 1.0) / (e_2x_6 + 1.0)
        S6 = tl.sin(beta_val * B5 + phi_val)
        B6 = T6 + alpha_val * S6
        B6_bf16 = B6.to(tl.bfloat16)
    if CHAOS_DEPTH >= 7:
        if LOAD_CHAOS:
            cs_offs = bh_id * stride_cs_bh + 6 * stride_cs_i + offs_tok[:, None] * D + offs_d[None, :]
            B6 = tl.load(ChaosStore_B + cs_offs)
            B6_bf16 = B6.to(tl.bfloat16)
            T7 = tl.load(ChaosStore_T + cs_offs)
        else:
            e_2x_7 = tl.exp(2.0 * tl.dot(B6_bf16, W_bf16, out_dtype=tl.float32))
            T7 = (e_2x_7 - 1.0) / (e_2x_7 + 1.0)
        S7 = tl.sin(beta_val * B6 + phi_val)
        B7 = T7 + alpha_val * S7

    # Select B_final as the last computed iterate (depth=0 -> A_i itself).
    if CHAOS_DEPTH == 0:
        B_final = B0
    elif CHAOS_DEPTH == 1:
        B_final = B1
    elif CHAOS_DEPTH == 2:
        B_final = B2
    elif CHAOS_DEPTH == 3:
        B_final = B3
    elif CHAOS_DEPTH == 4:
        B_final = B4
    elif CHAOS_DEPTH == 5:
        B_final = B5
    elif CHAOS_DEPTH == 6:
        B_final = B6
    else:
        B_final = B7

    C = 2.533 * A_i
    M = gate_0 * A_i + gate_1 * B_final + gate_2 * C

    dO_ptr = dO + bh_id * stride_h
    dO_i = tl.load(dO_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    dM = dO_i * (1.0 + perturb * tl.cos(M))
    dPerturb_local = tl.sum(dO_i * tl.sin(M))

    dA_mixer = gate_0 * dM
    dB_final = gate_1 * dM
    dC = gate_2 * dM
    dA_fractal = 2.533 * dC

    dGate0_local = tl.sum(dM * A_i)
    dGate1_local = tl.sum(dM * B_final)
    dGate2_local = tl.sum(dM * C)

    dW_local = tl.zeros([D, D], dtype=tl.float32)
    dAlpha_local = tl.zeros((), dtype=tl.float32)
    dBeta_local = tl.zeros((), dtype=tl.float32)
    dPhi_local = tl.zeros((), dtype=tl.float32)

    # Reverse-mode: dB is the gradient w.r.t. the current B_{i+1}. Each
    # constexpr-gated block backs up one chaos iteration.
    dB = dB_final

    if CHAOS_DEPTH >= 7:
        dZ = dB * (1.0 - T7 * T7)
        dArg = dB * (alpha_val * tl.cos(beta_val * B6 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B6_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S7)
        dBeta_local += tl.sum(dArg * B6)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 6:
        dZ = dB * (1.0 - T6 * T6)
        dArg = dB * (alpha_val * tl.cos(beta_val * B5 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B5_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S6)
        dBeta_local += tl.sum(dArg * B5)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 5:
        dZ = dB * (1.0 - T5 * T5)
        dArg = dB * (alpha_val * tl.cos(beta_val * B4 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B4_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S5)
        dBeta_local += tl.sum(dArg * B4)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 4:
        dZ = dB * (1.0 - T4 * T4)
        dArg = dB * (alpha_val * tl.cos(beta_val * B3 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B3_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S4)
        dBeta_local += tl.sum(dArg * B3)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 3:
        dZ = dB * (1.0 - T3 * T3)
        dArg = dB * (alpha_val * tl.cos(beta_val * B2 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B2_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S3)
        dBeta_local += tl.sum(dArg * B2)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 2:
        dZ = dB * (1.0 - T2 * T2)
        dArg = dB * (alpha_val * tl.cos(beta_val * B1 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B1_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S2)
        dBeta_local += tl.sum(dArg * B1)
        dPhi_local += tl.sum(dArg)
    if CHAOS_DEPTH >= 1:
        dZ = dB * (1.0 - T1 * T1)
        dArg = dB * (alpha_val * tl.cos(beta_val * B0 + phi_val))
        dB_in = dB
        dB = tl.dot(dZ.to(tl.bfloat16), WT_bf16, out_dtype=tl.float32) + beta_val * dArg
        dW_local += tl.dot(tl.trans(B0_bf16), dZ.to(tl.bfloat16), out_dtype=tl.float32)
        dAlpha_local += tl.sum(dB_in * S1)
        dBeta_local += tl.sum(dArg * B0)
        dPhi_local += tl.sum(dArg)

    dB0 = dB  # gradient w.r.t. A via the chaos path (== dB_final when depth=0)
    dA_total = dA_mixer + dA_fractal + dB0
    dA_total = tl.where(q_mask[:, None], dA_total, 0.0)

    dA_ptr = dA_total_out + bh_id * stride_h
    tl.store(dA_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], dA_total.to(tl.bfloat16))

    # Di = sum_d(A_i[d] * dA_total[d]) is the row-sum the attn bwd needs to
    # assemble dS. Computing it here (A_i is already in registers) lets the
    # attn bwd skip an entire extra K-loop over the sparsity pattern.
    Di_local = tl.sum(A_i * dA_total, axis=1)
    Di_local = tl.where(q_mask, Di_local, 0.0)
    tl.store(Di_out + bh_id * stride_lh + offs_tok, Di_local, mask=q_mask)

    W_offs_x = tl.arange(0, D)[:, None]
    W_offs_y = tl.arange(0, D)[None, :]
    tl.atomic_add(dW_workspace + W_offs_x * D + W_offs_y, dW_local)

    tl.atomic_add(dScalars_workspace + 0, dAlpha_local)
    tl.atomic_add(dScalars_workspace + 1, dBeta_local)
    tl.atomic_add(dScalars_workspace + 2, dPhi_local)

    tl.atomic_add(dMixer_workspace + 0, dGate0_local)
    tl.atomic_add(dMixer_workspace + 1, dGate1_local)
    tl.atomic_add(dMixer_workspace + 2, dGate2_local)

    tl.atomic_add(dPerturb_workspace + 0, dPerturb_local)



@triton.jit
def _vortex_attn_bwd_dq_kernel(
    Q, K, V, LSE,
    dA_total,
    Di_in,
    dQ,
    RowPtr, ColIdx, SeqLens,
    stride_cih,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_rph: tl.constexpr = (T_MAX // BS) + 1
    stride_lh: tl.constexpr = T_MAX
    NUM_Q_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    seq_len = tl.load(SeqLens + bh_id // NUM_HEADS)
    q_start = q_block_id * BS
    if q_start >= seq_len:
        return

    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)
    q_mask = offs_tok < seq_len

    Q_ptr = Q + bh_id * stride_h
    q_bf16 = tl.load(Q_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    dA_ptr = dA_total + bh_id * stride_h
    dA_i = tl.load(dA_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    LSE_ptr = LSE + bh_id * stride_lh
    LSE_i = tl.load(LSE_ptr + offs_tok)

    rp_base = RowPtr + bh_id * stride_rph + q_block_id
    ci_lo = tl.load(rp_base)
    ci_hi = tl.load(rp_base + 1)

    K_ptr = K + bh_id * stride_h
    V_ptr = V + bh_id * stride_h
    CI_ptr = ColIdx + bh_id * stride_cih

    offs_k_tile = tl.arange(0, BS)
    boundary = (q_start + BS) > seq_len

    Di = tl.load(Di_in + bh_id * stride_lh + offs_tok, mask=q_mask, other=0.0)

    dQ_i = tl.zeros([BS, D], dtype=tl.float32)

    if ci_hi > ci_lo:
        k_start_d = q_block_id * BS
        offs_k_d = k_start_d + offs_k_tile
        if boundary:
            k_mask_d = offs_k_d < seq_len
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask_d[:, None], other=0.0)
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask_d[:, None], other=0.0)
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            s_d = tl.where(causal & k_mask_d[None, :], s_d, float("-inf"))
        else:
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            s_d = tl.where(causal, s_d, float("-inf"))

        P_d = tl.exp(s_d - LSE_i[:, None])
        dP_d = tl.dot(dA_i, tl.trans(v_bf16_d), out_dtype=tl.float32)
        dS_d = (P_d * (dP_d - Di[:, None])) * SCALE
        dQ_i += tl.dot(dS_d.to(tl.bfloat16), k_bf16_d, out_dtype=tl.float32)

    for ci in range(ci_lo, ci_hi - 1):
        k_block_id = tl.load(CI_ptr + ci)
        k_start = k_block_id * BS
        offs_k = k_start + offs_k_tile
        k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
        v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
        s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
        P = tl.exp(s - LSE_i[:, None])
        dP = tl.dot(dA_i, tl.trans(v_bf16), out_dtype=tl.float32)
        dS = (P * (dP - Di[:, None])) * SCALE
        dQ_i += tl.dot(dS.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)

    dQ_ptr = dQ + bh_id * stride_h
    tl.store(dQ_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], dQ_i.to(tl.bfloat16), mask=q_mask[:, None])


@triton.jit
def _vortex_attn_bwd_dkv_kernel(
    Q, K, V, LSE,
    dA_total,
    Di_in,
    dK, dV,
    RowPtrT, ColIdxT, SeqLens,
    stride_cith,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_rpth: tl.constexpr = (T_MAX // BS) + 1
    stride_lh: tl.constexpr = T_MAX
    NUM_K_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_K_BLOCKS
    k_block_id = pid % NUM_K_BLOCKS

    seq_len = tl.load(SeqLens + bh_id // NUM_HEADS)
    k_start = k_block_id * BS
    if k_start >= seq_len:
        return

    offs_d = tl.arange(0, D)
    offs_tok_k = k_start + tl.arange(0, BS)
    k_mask = offs_tok_k < seq_len

    K_ptr = K + bh_id * stride_h
    V_ptr = V + bh_id * stride_h
    Q_ptr = Q + bh_id * stride_h
    dA_ptr = dA_total + bh_id * stride_h
    LSE_ptr = LSE + bh_id * stride_lh

    k_bf16 = tl.load(K_ptr + offs_tok_k[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)
    v_bf16 = tl.load(V_ptr + offs_tok_k[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)

    dK_acc = tl.zeros([BS, D], dtype=tl.float32)
    dV_acc = tl.zeros([BS, D], dtype=tl.float32)

    # Diagonal: Q-block k_block_id attending to K-block k_block_id (causal).
    q_start_diag = k_block_id * BS
    offs_tok_q = q_start_diag + tl.arange(0, BS)
    q_mask_diag = offs_tok_q < seq_len
    q_bf16_d = tl.load(Q_ptr + offs_tok_q[:, None] * stride_t + offs_d[None, :],
                       mask=q_mask_diag[:, None], other=0.0)
    dA_d = tl.load(dA_ptr + offs_tok_q[:, None] * stride_t + offs_d[None, :],
                   mask=q_mask_diag[:, None], other=0.0)
    LSE_d = tl.load(LSE_ptr + offs_tok_q, mask=q_mask_diag, other=0.0)
    Di_d = tl.load(Di_in + bh_id * stride_lh + offs_tok_q, mask=q_mask_diag, other=0.0)

    s_d = tl.dot(q_bf16_d, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
    causal_d = offs_tok_q[:, None] >= offs_tok_k[None, :]
    s_d = tl.where(causal_d & k_mask[None, :] & q_mask_diag[:, None], s_d, float("-inf"))
    P_d = tl.exp(s_d - LSE_d[:, None])
    dP_d = tl.dot(dA_d, tl.trans(v_bf16), out_dtype=tl.float32)
    dS_d = (P_d * (dP_d - Di_d[:, None])) * SCALE

    dK_acc += tl.dot(tl.trans(dS_d.to(tl.bfloat16)), q_bf16_d, out_dtype=tl.float32)
    dV_acc += tl.dot(tl.trans(P_d.to(tl.bfloat16)), dA_d, out_dtype=tl.float32)

    # Off-diagonal: iterate Q-blocks that attend to this K-block via the
    # transposed CSR.
    rpt_base = RowPtrT + bh_id * stride_rpth + k_block_id
    cit_lo = tl.load(rpt_base)
    cit_hi = tl.load(rpt_base + 1)
    CIT_ptr = ColIdxT + bh_id * stride_cith

    for ci in range(cit_lo, cit_hi):
        q_block_id = tl.load(CIT_ptr + ci)
        q_start = q_block_id * BS
        offs_tok_q2 = q_start + tl.arange(0, BS)
        q_mask2 = offs_tok_q2 < seq_len
        q_bf16_o = tl.load(Q_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :],
                           mask=q_mask2[:, None], other=0.0)
        dA_o = tl.load(dA_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :],
                       mask=q_mask2[:, None], other=0.0)
        LSE_o = tl.load(LSE_ptr + offs_tok_q2, mask=q_mask2, other=0.0)
        Di_o = tl.load(Di_in + bh_id * stride_lh + offs_tok_q2, mask=q_mask2, other=0.0)

        s_o = tl.dot(q_bf16_o, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
        s_o = tl.where(q_mask2[:, None] & k_mask[None, :], s_o, float("-inf"))
        P_o = tl.exp(s_o - LSE_o[:, None])
        dP_o = tl.dot(dA_o, tl.trans(v_bf16), out_dtype=tl.float32)
        dS_o = (P_o * (dP_o - Di_o[:, None])) * SCALE

        dK_acc += tl.dot(tl.trans(dS_o.to(tl.bfloat16)), q_bf16_o, out_dtype=tl.float32)
        dV_acc += tl.dot(tl.trans(P_o.to(tl.bfloat16)), dA_o, out_dtype=tl.float32)

    dK_ptr = dK + bh_id * stride_h + offs_tok_k[:, None] * stride_t + offs_d[None, :]
    dV_ptr = dV + bh_id * stride_h + offs_tok_k[:, None] * stride_t + offs_d[None, :]
    tl.store(dK_ptr, dK_acc.to(tl.bfloat16), mask=k_mask[:, None])
    tl.store(dV_ptr, dV_acc.to(tl.bfloat16), mask=k_mask[:, None])



def launch_vortex_fused_bwd(
    q, k, v,
    proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
    do, lse,
    row_ptr, col_idx, seq_lens,
    chaos_store_B=None, chaos_store_T=None,
    a_store=None,
    row_ptr_T=None, col_idx_T=None,
):
    qshape = q.shape
    batch_size = qshape[0]
    num_heads = qshape[1]
    t_max = qshape[2]
    batch_heads = batch_size * num_heads
    num_q_blocks = t_max // BLOCK_SIZE

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da_total = torch.empty_like(q)

    dw_workspace = torch.zeros_like(proj_weight, dtype=torch.float32)
    dscalars_workspace = torch.zeros_like(chaos_scalars, dtype=torch.float32)
    dmixer_workspace = torch.zeros_like(mixer_gate, dtype=torch.float32)
    dperturb_workspace = torch.zeros_like(chaos_perturb, dtype=torch.float32)

    # Di = row-sum of A_i * dA_i, computed in chaos_bwd from in-register A_i
    # and reused by attn_bwd so it does not have to re-run the sparsity loop.
    di_scratch = torch.empty((batch_heads, t_max), device=q.device, dtype=torch.float32)

    # Determine whether the fwd populated the scratch. If not, pass 1-elem
    # placeholders; the kernel never dereferences them when LOAD_CHAOS=0.
    load_chaos = bool(CHAOS_STORE and chaos_store_B is not None
                      and chaos_store_B.numel() > 1)
    if not load_chaos:
        chaos_store_B = torch.empty(1, device=q.device, dtype=torch.float32)
        chaos_store_T = torch.empty(1, device=q.device, dtype=torch.float32)

    load_a = bool(a_store is not None and a_store.numel() > 1)
    if not load_a:
        a_store = torch.empty(1, device=q.device, dtype=torch.float32)

    # Transposed CSR for K-parallel dK/dV. Builds on demand if the caller did
    # not pre-compute one (cheap for typical shapes, but callers should cache).
    if row_ptr_T is None or col_idx_T is None:
        row_ptr_T, col_idx_T = build_attn_bwd_csrT(row_ptr, col_idx, num_q_blocks)

    grid = (num_q_blocks * batch_heads,)
    stride_cih = col_idx.shape[2] if col_idx.ndim == 3 else col_idx.shape[1]
    stride_cith = col_idx_T.shape[2] if col_idx_T.ndim == 3 else col_idx_T.shape[1]

    if _BWD_PROFILE:
        ev_c0 = torch.cuda.Event(enable_timing=True); ev_c1 = torch.cuda.Event(enable_timing=True)
        ev_a0 = torch.cuda.Event(enable_timing=True); ev_a1 = torch.cuda.Event(enable_timing=True)
        ev_c0.record()
    _vortex_chaos_bwd_kernel[grid](
        q, k, v,
        proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
        do,
        da_total,
        di_scratch,
        dw_workspace, dscalars_workspace, dmixer_workspace, dperturb_workspace,
        row_ptr, col_idx, seq_lens,
        chaos_store_B, chaos_store_T,
        a_store,
        stride_cih,
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        SCALE=1.0 / math.sqrt(HEAD_DIM),
        BS=BLOCK_SIZE,
        D=HEAD_DIM,
        CHAOS_DEPTH=CHAOS_DEPTH,
        LOAD_CHAOS=int(load_chaos),
        LOAD_A=int(load_a),
        num_stages=_BWD_CHAOS_NUM_STAGES,
        num_warps=_BWD_CHAOS_NUM_WARPS,
    )
    if _BWD_PROFILE:
        ev_c1.record()
        ev_dq0 = torch.cuda.Event(enable_timing=True); ev_dq1 = torch.cuda.Event(enable_timing=True)
        ev_dkv0 = torch.cuda.Event(enable_timing=True); ev_dkv1 = torch.cuda.Event(enable_timing=True)
        ev_dq0.record()
    _vortex_attn_bwd_dq_kernel[grid](
        q, k, v, lse,
        da_total,
        di_scratch,
        dq,
        row_ptr, col_idx, seq_lens,
        stride_cih,
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        SCALE=1.0 / math.sqrt(HEAD_DIM),
        BS=BLOCK_SIZE,
        D=HEAD_DIM,
        num_stages=_BWD_ATTN_NUM_STAGES,
        num_warps=_BWD_ATTN_NUM_WARPS,
    )
    if _BWD_PROFILE:
        ev_dq1.record(); ev_dkv0.record()
    _vortex_attn_bwd_dkv_kernel[grid](
        q, k, v, lse,
        da_total,
        di_scratch,
        dk, dv,
        row_ptr_T, col_idx_T, seq_lens,
        stride_cith,
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        SCALE=1.0 / math.sqrt(HEAD_DIM),
        BS=BLOCK_SIZE,
        D=HEAD_DIM,
        num_stages=_BWD_ATTN_DKV_NUM_STAGES,
        num_warps=_BWD_ATTN_DKV_NUM_WARPS,
    )
    if _BWD_PROFILE:
        ev_dkv1.record(); torch.cuda.synchronize()
        print(f"[bwd_profile] chaos={ev_c0.elapsed_time(ev_c1):.3f}ms  "
              f"attn_dq={ev_dq0.elapsed_time(ev_dq1):.3f}ms  "
              f"attn_dkv={ev_dkv0.elapsed_time(ev_dkv1):.3f}ms", flush=True)

    # Apply softmax Jacobian to dmixer grads accumulated in pre-softmax space.
    with torch.no_grad():
        gate = torch.nn.functional.softmax(mixer_gate, dim=0).float()
        dg_hat = dmixer_workspace
        dg = torch.zeros_like(mixer_gate)
        dot = dg_hat[0] * gate[0] + dg_hat[1] * gate[1] + dg_hat[2] * gate[2]
        dg[0] = gate[0] * (dg_hat[0] - dot)
        dg[1] = gate[1] * (dg_hat[1] - dot)
        dg[2] = gate[2] * (dg_hat[2] - dot)

    return dq, dk, dv, dw_workspace.to(q.dtype), dscalars_workspace, dg, dperturb_workspace
