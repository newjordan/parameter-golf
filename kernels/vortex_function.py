import torch
import math
from kernels.vortex_fused import launch_vortex_fused, BLOCK_SIZE
from kernels.vortex_bwd import launch_vortex_fused_bwd, build_attn_bwd_csrT

_CSRT_CACHE = {}


def _get_csrT(row_ptr, col_idx, num_blocks):
    key = (row_ptr.data_ptr(), col_idx.data_ptr(), num_blocks,
           tuple(row_ptr.shape), tuple(col_idx.shape))
    hit = _CSRT_CACHE.get(key)
    if hit is not None:
        return hit
    rp_T, ci_T = build_attn_bwd_csrT(row_ptr, col_idx, num_blocks)
    _CSRT_CACHE[key] = (rp_T, ci_T)
    return rp_T, ci_T


class VortexHelixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, row_ptr, col_idx, seq_lens):
        o, lse_3d, chaos_store_B, chaos_store_T, a_store = launch_vortex_fused(
            q, k, v,
            proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
            row_ptr, col_idx, seq_lens
        )
        ctx.save_for_backward(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, lse_3d, row_ptr, col_idx, seq_lens)
        # Scratch tensors are not leaves; stash on ctx directly (non-tensor attr).
        ctx.chaos_store_B = chaos_store_B
        ctx.chaos_store_T = chaos_store_T
        ctx.a_store = a_store
        num_q_blocks = q.shape[2] // BLOCK_SIZE
        ctx.row_ptr_T, ctx.col_idx_T = _get_csrT(row_ptr, col_idx, num_q_blocks)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, lse_3d, row_ptr, col_idx, seq_lens = ctx.saved_tensors

        dq, dk, dv, dw, dscalars, dmux, dperturb = launch_vortex_fused_bwd(
            q, k, v,
            proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
            do, lse_3d,
            row_ptr, col_idx, seq_lens,
            chaos_store_B=ctx.chaos_store_B,
            chaos_store_T=ctx.chaos_store_T,
            a_store=ctx.a_store,
            row_ptr_T=ctx.row_ptr_T,
            col_idx_T=ctx.col_idx_T,
        )

        return dq, dk, dv, dw, dscalars, dmux, dperturb, None, None, None
