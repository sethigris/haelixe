// --------------------------------------------------------------------------
// Module: kernels::softmax
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Implements a numerically stable Softmax operation and its exact
//   Jacobian backward pass. Operates strictly over the last dimension
//   of the tensor, which is the exact memory layout required for
//   LLM Attention scores [Batch, Heads, Seq, Seq].
//
// MATHEMATICAL INVARIANTS:
//   - Forward: Uses the LogSumExp trick. y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x))).
//     This prevents exp() from overflowing to Infinity.
//   - Backward: The Jacobian of Softmax depends on the OUTPUT (y), not
//     the input (x). dx_i = y_i * (grad_i - sum(grad_j * y_j)).
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-22
// --------------------------------------------------------------------------
use crate::{Tensor, DType};
use rayon::prelude::*;

/// Computes the numerically stable Softmax over the last dimension.
pub fn softmax_forward(x: &Tensor) -> Tensor {
    let x_cpu = x.ensure_cpu();
    let dims = x_cpu.shape.dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = x_cpu.shape.num_elements() / last_dim;

    let x_addr = x_cpu.storage.as_ptr() as *const f32 as usize;
    let mut out_data = vec![0.0f32; x_cpu.shape.num_elements()];
    let out_addr = out_data.as_mut_ptr() as usize;

    (0..batch_size).into_par_iter().for_each(|b| {
        unsafe {
            let row_start = b * last_dim;
            let x_row = (x_addr as *const f32).add(row_start);
            let out_row = (out_addr as *mut f32).add(row_start);

            // 1. Find Max (The LogSumExp Trick)
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..last_dim {
                let val = *x_row.add(i);
                if val > max_val { max_val = val; }
            }

            // 2. Exp and Sum
            let mut sum_exp = 0.0f32;
            for i in 0..last_dim {
                let val = (*x_row.add(i) - max_val).exp();
                *out_row.add(i) = val;
                sum_exp += val;
            }

            // 3. Normalize
            let inv_sum = 1.0 / sum_exp;
            for i in 0..last_dim {
                *out_row.add(i) *= inv_sum;
            }
        }
    });
    Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &out_data)
}

/// Computes the exact Jacobian backward pass for Softmax.
/// CRITICAL: Requires the OUTPUT of the forward pass (y), not the input (x).
pub fn softmax_backward(y: &Tensor, grad: &Tensor) -> Tensor {
    let y_cpu = y.ensure_cpu();
    let g_cpu = grad.ensure_cpu();
    let dims = y_cpu.shape.dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = y_cpu.shape.num_elements() / last_dim;

    // Cast raw pointers to usize to bypass Rayon's Sync trait paranoia.
    let y_addr = y_cpu.storage.as_ptr() as *const f32 as usize;
    let g_addr = g_cpu.storage.as_ptr() as *const f32 as usize;
    let mut dx_data = vec![0.0f32; y_cpu.shape.num_elements()];
    let dx_addr = dx_data.as_mut_ptr() as usize;

    (0..batch_size).into_par_iter().for_each(|b| {
        unsafe {
            let offset = b * last_dim;
            let y_row = (y_addr as *const f32).add(offset);
            let g_row = (g_addr as *const f32).add(offset);
            let dx_row = (dx_addr as *mut f32).add(offset);

            // dot = sum(grad * y)
            let mut dot = 0.0f32;
            for i in 0..last_dim {
                dot += *g_row.add(i) * *y_row.add(i);
            }

            // dx_i = y_i * (grad_i - dot)
            for i in 0..last_dim {
                *dx_row.add(i) = *y_row.add(i) * (*g_row.add(i) - dot);
            }
        }
    });
    Tensor::from_slice(DType::F32, y_cpu.shape.clone(), &dx_data)
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
// This implementation is strictly CPU-bound and iterates row-by-row.
// While mathematically pure and numerically stable, it will bottleneck
// on massive sequence lengths (e.g., 32k context windows). The engineer
// who inherits this file must replace this Rayon loop with a WGSL
// compute shader utilizing Workgroup Shared Memory (SRAM) reductions.
// --------------------------------------------------------------------------
