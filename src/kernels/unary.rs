// --------------------------------------------------------------------------
// Module: kernels::unary
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Executes unary element-wise operations on the CPU fallback path
//   and computes the exact mathematical derivatives for the backward
//   pass of the Autograd graph.
//
// HISTORICAL CONTEXT:
//   Forged alongside the WGSL unary shader to guarantee mathematical
//   parity between GPU and CPU execution paths.
//
// INVARIANTS:
//   - The backward pass strictly applies the chain rule: dx = grad * f'(x).
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-22
// --------------------------------------------------------------------------

use crate::{DType, Tensor};
use rayon::prelude::*;

pub fn forward_cpu(x: &Tensor, op_code: u32) -> Tensor {
    let x_cpu = x.ensure_cpu();
    let n = x_cpu.shape.num_elements();
    let mut out = vec![0.0f32; n];
    let x_ptr = x_cpu.storage.as_ptr() as *const f32 as usize;
    let out_ptr = out.as_mut_ptr() as usize;

    (0..n).into_par_iter().for_each(|i| unsafe {
        let val = *((x_ptr as *const f32).add(i));
        let res = match op_code {
            0 => val.exp(),
            1 => val.ln(),
            2 => val.sqrt(),
            3 => val.tanh(),
            4 => val.max(0.0),
            5 => {
                // GELU
                let c = 0.7978845608;
                let inner = c * (val + 0.044715 * val.powi(3));
                0.5 * val * (1.0 + inner.tanh())
            }
            6 => {
                // SiLU
                let sig = 1.0 / (1.0 + (-val).exp());
                val * sig
            }
            _ => 0.0,
        };
        *((out_ptr as *mut f32).add(i)) = res;
    });
    Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &out)
}

pub fn backward_cpu(x: &Tensor, grad: &Tensor, op_code: u32) -> Tensor {
    let x_cpu = x.ensure_cpu();
    let g_cpu = grad.ensure_cpu();
    let n = x_cpu.shape.num_elements();
    let mut dx = vec![0.0f32; n];
    let x_ptr = x_cpu.storage.as_ptr() as *const f32 as usize;
    let g_ptr = g_cpu.storage.as_ptr() as *const f32 as usize;
    let dx_ptr = dx.as_mut_ptr() as usize;

    (0..n).into_par_iter().for_each(|i| unsafe {
        let val = *((x_ptr as *const f32).add(i));
        let g = *((g_ptr as *const f32).add(i));

        // Mathematical Derivatives: dx = grad * f'(x)
        let deriv = match op_code {
            0 => val.exp(),                // d/dx exp(x)
            1 => 1.0 / val,                // d/dx ln(x)
            2 => 0.5 / val.sqrt(),         // d/dx sqrt(x)
            3 => 1.0 - val.tanh().powi(2), // d/dx tanh(x)
            4 => {
                if val > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } // d/dx ReLU
            5 => {
                // GELU (Tanh approx derivative)
                let c = 0.7978845608;
                let inner = c * (val + 0.044715 * val.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner.powi(2);
                let d_inner = c * (1.0 + 3.0 * 0.044715 * val.powi(2));
                0.5 * (1.0 + tanh_inner) + 0.5 * val * sech2 * d_inner
            }
            6 => {
                // SiLU derivative
                let sig = 1.0 / (1.0 + (-val).exp());
                sig + val * sig * (1.0 - sig)
            }
            _ => 0.0,
        };
        *((dx_ptr as *mut f32).add(i)) = g * deriv;
    });
    Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &dx)
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
// The GELU derivative implemented here uses the exact derivative of the
// Tanh approximation, not the exact derivative of the true GELU (which
// requires the Cumulative Distribution Function). While this guarantees
// perfect forward/backward consistency for the approximation, models
// requiring the exact CDF-based GELU will need a separate op_code.
// --------------------------------------------------------------------------
