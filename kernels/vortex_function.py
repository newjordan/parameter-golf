import torch
import math
from kernels.vortex_fused import launch_vortex_fused
from kernels.vortex_bwd import launch_vortex_fused_bwd

class VortexHelixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, row_ptr, col_idx, seq_lens):
        o, lse_3d, chaos_store_B, chaos_store_T = launch_vortex_fused(
            q, k, v,
            proj_weight, chaos_scalars, mixer_gate, chaos_perturb,
            row_ptr, col_idx, seq_lens
        )
        ctx.save_for_backward(q, k, v, proj_weight, chaos_scalars, mixer_gate, chaos_perturb, lse_3d, row_ptr, col_idx, seq_lens)
        # Scratch tensors are not leaves; stash on ctx directly (non-tensor attr).
        ctx.chaos_store_B = chaos_store_B
        ctx.chaos_store_T = chaos_store_T
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
        )

        return dq, dk, dv, dw, dscalars, dmux, dperturb, None, None, None
