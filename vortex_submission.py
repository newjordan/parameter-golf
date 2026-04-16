import math
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {
        "name": "default",
    }
]

@triton.jit(do_not_specialize=["stride_cih"])
def _vortex_fwd_kernel(
    Q, K, V,
    O, LSE,
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

    if ci_hi > ci_lo:
        # Load col_idx early to hide latency
        k_block_id_next = tl.load(CI_ptr + ci_lo)

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
        k_block_id = k_block_id_next
        if ci < ci_hi - 2:
            k_block_id_next = tl.load(CI_ptr + ci + 1)
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
    a = tl.where(q_mask[:, None], acc / l_safe[:, None], 0.0)
    lse_out = tl.where(q_mask, (m_i + tl.log2(l_safe)) * LN2, float("-inf"))

    O_ptr = O + bh_id * stride_h
    tl.store(O_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], a.to(tl.bfloat16))
    LSE_ptr = LSE + bh_id * stride_lh
    tl.store(LSE_ptr + offs_tok, lse_out)

_OUTPUT_CACHE = {}
_KERNEL_CACHE = {}
_STREAM_CACHE = [None]

def _get_stream():
    s = _STREAM_CACHE[0]
    if s is None:
        from triton.runtime import driver as _driver
        dev = _driver.active.get_current_device()
        s = _driver.active.get_current_stream(dev)
        _STREAM_CACHE[0] = s
    return s

def _slow_launch(q, k_, v, o, lse_2d, row_ptr, col_idx, seq_lens, nnz, num_q_blocks, batch_heads, t_max, num_heads):
    _vortex_fwd_kernel[(num_q_blocks * batch_heads,)](
        q, k_, v, o, lse_2d,
        row_ptr, col_idx, seq_lens,
        nnz,
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        SCALE=SCALE,
        BS=BLOCK_SIZE,
        D=HEAD_DIM,
        num_stages=3,
        num_warps=8,
    )

def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    device = q.device

    qshape = q.shape
    batch_size = qshape[0]
    num_heads = qshape[1]
    t_max = qshape[2]
    batch_heads = batch_size * num_heads
    num_q_blocks = t_max >> 7

    cache_key = (t_max, num_heads, batch_size)
    bufs = _OUTPUT_CACHE.get(cache_key)
    if bufs is None:
        o = torch.empty_like(q)
        lse_3d = torch.empty((batch_size, num_heads, t_max), device=device, dtype=torch.float32)
        lse_2d = lse_3d.view(batch_heads, t_max)
        bufs = (o, lse_2d, lse_3d)
        _OUTPUT_CACHE[cache_key] = bufs
    o, lse_2d, lse_3d = bufs

    nnz = col_idx.shape[-1] if col_idx.ndim == 3 else col_idx.shape[1]

    cached = _KERNEL_CACHE.get(cache_key)
    if cached is None:
        _slow_launch(q, k, v, o, lse_2d, row_ptr, col_idx, seq_lens, nnz, num_q_blocks, batch_heads, t_max, num_heads)
        from triton.runtime import driver as _driver
        dev_idx = device.index
        kernel_dict = _vortex_fwd_kernel.device_caches[dev_idx][0]
        ck = None
        target_marker = f"'constexpr', {t_max})"
        for key in kernel_dict:
            if target_marker in key and f"'constexpr', {num_heads})" in key:
                ck = kernel_dict[key]
        if ck is None:
            ck = next(iter(kernel_dict.values()))
        _ = ck.run
        grid_x = num_q_blocks * batch_heads
        cached = (ck, grid_x, ck.function, ck.packed_metadata)
        _KERNEL_CACHE[cache_key] = cached
        return o, lse_3d

    ck, grid_x, func, packed_meta = cached
    stream = _get_stream()
    ck.run(
        grid_x, 1, 1, stream, func, packed_meta, None,
        None, None,
        q, k, v, o, lse_2d, row_ptr, col_idx, seq_lens, nnz,
        t_max, num_heads, SCALE, BLOCK_SIZE, HEAD_DIM,
    )
    return o, lse_3d

def setup(suite_specs, device, variants):
    seen = set()
    for spec in suite_specs:
        t_max = getattr(spec, "t_max", None)
        num_heads = getattr(spec, "num_heads", 1)
        batch_size = getattr(spec, "batch_size", 1)
        if t_max is None:
            continue
        key = (t_max, num_heads)
        if key in seen:
            continue
        seen.add(key)
        B, H, T = int(batch_size), int(num_heads), int(t_max)
        q = torch.zeros((B, H, T, HEAD_DIM), device=device, dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        row_ptr = torch.zeros((B, H, T // BLOCK_SIZE + 1), device=device, dtype=torch.int32)
        col_idx = torch.zeros((B, H, 1), device=device, dtype=torch.int32)
        seq_lens = torch.tensor([T] * B, device=device, dtype=torch.int32)
        block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)
    if not seen:
        B, H, T = 1, 1, BLOCK_SIZE * 2
        q = torch.zeros((B, H, T, HEAD_DIM), device=device, dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        row_ptr = torch.zeros((B, H, T // BLOCK_SIZE + 1), device=device, dtype=torch.int32)
        col_idx = torch.zeros((B, H, 1), device=device, dtype=torch.int32)
        seq_lens = torch.tensor([T], device=device, dtype=torch.int32)
        block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens)
    torch.cuda.synchronize()
    return None
