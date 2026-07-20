// --------------------------------------------------------------------------
// Module: kernels::rms_norm
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Fused bare-metal CPU kernel for Root Mean Square Layer Normalization.
//   By fusing the square, mean, rsqrt, and scale operations into a single
//   pass, we eliminate intermediate tensor allocations and autograd graph
//   bloat, preserving VRAM and memory bandwidth.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use crate::{DType, Tensor};
use rayon::prelude::*;
use std::sync::Mutex;

pub fn rms_norm_forward(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let x_cpu = x.ensure_cpu();
    let w_cpu = weight.ensure_cpu();

    let dims = x_cpu.shape.dims();
    let hidden_dim = *dims.last().unwrap();
    let batch_size = x_cpu.shape.num_elements() / hidden_dim;

    let x_addr = x_cpu.storage.as_ptr() as *const f32 as usize;
    let w_addr = w_cpu.storage.as_ptr() as *const f32 as usize;

    let mut out_data = vec![0.0f32; x_cpu.shape.num_elements()];
    let out_addr = out_data.as_mut_ptr() as usize;

    (0..batch_size).into_par_iter().for_each(|i| unsafe {
        let x_row = (x_addr as *const f32).add(i * hidden_dim);
        let out_row = (out_addr as *mut f32).add(i * hidden_dim);
        let w = w_addr as *const f32;

        let mut sum_sq = 0.0f32;
        for j in 0..hidden_dim {
            let val = *x_row.add(j);
            sum_sq += val * val;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();

        for j in 0..hidden_dim {
            let val = *x_row.add(j);
            let weight_val = *w.add(j);
            *out_row.add(j) = val * inv_rms * weight_val;
        }
    });

    Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &out_data)
}

pub fn rms_norm_backward(
    grad_out: &Tensor,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> (Tensor, Tensor) {
    let g_cpu = grad_out.ensure_cpu();
    let x_cpu = x.ensure_cpu();
    let w_cpu = weight.ensure_cpu();

    let dims = x_cpu.shape.dims();
    let hidden_dim = *dims.last().unwrap();
    let batch_size = x_cpu.shape.num_elements() / hidden_dim;

    let g_addr = g_cpu.storage.as_ptr() as *const f32 as usize;
    let x_addr = x_cpu.storage.as_ptr() as *const f32 as usize;
    let w_addr = w_cpu.storage.as_ptr() as *const f32 as usize;

    let mut dx_data = vec![0.0f32; x_cpu.shape.num_elements()];
    let dx_addr = dx_data.as_mut_ptr() as usize;

    let dw_mutex = Mutex::new(vec![0.0f32; hidden_dim]);

    (0..batch_size).into_par_iter().for_each(|i| {
        unsafe {
            let g_row = (g_addr as *const f32).add(i * hidden_dim);
            let x_row = (x_addr as *const f32).add(i * hidden_dim);
            let w = w_addr as *const f32;
            let dx_row = (dx_addr as *mut f32).add(i * hidden_dim);

            // Gradient Checkpointing: Recompute forward stats to save memory
            let mut sum_sq = 0.0f32;
            for j in 0..hidden_dim {
                let val = *x_row.add(j);
                sum_sq += val * val;
            }
            let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();

            let mut dot = 0.0f32;
            for j in 0..hidden_dim {
                let g_val = *g_row.add(j);
                let x_val = *x_row.add(j);
                let w_val = *w.add(j);
                dot += g_val * w_val * x_val;
            }

            let mut local_dw = vec![0.0f32; hidden_dim];

            for j in 0..hidden_dim {
                let g_val = *g_row.add(j);
                let x_val = *x_row.add(j);
                let w_val = *w.add(j);

                let dx_val =
                    (g_val * w_val * inv_rms) - (x_val * inv_rms.powi(3) * dot / hidden_dim as f32);
                *dx_row.add(j) = dx_val;

                local_dw[j] = g_val * x_val * inv_rms;
            }

            let mut dw_lock = dw_mutex.lock().unwrap();
            for j in 0..hidden_dim {
                dw_lock[j] += local_dw[j];
            }
        }
    });

    let dx = Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &dx_data);
    let dw = Tensor::from_slice(
        DType::F32,
        w_cpu.shape.clone(),
        &dw_mutex.into_inner().unwrap(),
    );

    (dx, dw)
}
