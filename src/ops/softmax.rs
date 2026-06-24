use crate::{Op, Tensor};
use rayon::prelude::*;

#[derive(Debug)]
pub struct SoftmaxOp {
    pub output: Tensor, // We store the FORWARD output (the probabilities) to save memory!
}

impl Op for SoftmaxOp {
    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // The Jacobian of softmax is: S_i * (grad_i - sum(grad_j * S_j))
        // We can compute this elegantly without allocating a massive Jacobian matrix.
        let s = &self.output;
        let grad_out = grad_output;

        // 1. Element-wise multiply gradient by softmax probabilities
        // (We will use a simple CPU loop here for brevity, or you can write a kernel)
        let s_cpu = s.ensure_cpu();
        let g_cpu = grad_out.ensure_cpu();
        let out = Tensor::empty(s_cpu.dtype, s_cpu.shape.clone());

        let num_elements = s_cpu.shape.num_elements();
        let last_dim = s_cpu.shape.dims().last().copied().unwrap_or(1);
        let num_rows = num_elements / last_dim;

        let s_ptr = crate::kernels::activations::SyncPtr(s_cpu.storage.as_ptr() as *const f32);
        let g_ptr = crate::kernels::activations::SyncPtr(g_cpu.storage.as_ptr() as *const f32);
        let out_ptr = crate::kernels::activations::SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

        (0..num_rows).into_par_iter().for_each(|row_idx| {
            let row_start = row_idx * last_dim;

            // Calculate the dot product of grad and softmax for this row
            let mut dot = 0.0f32;
            for i in 0..last_dim {
                unsafe {
                    dot += *g_ptr.get().add(row_start + i) * *s_ptr.get().add(row_start + i);
                }
            }

            // Apply the Jacobian formula
            for i in 0..last_dim {
                unsafe {
                    let s_val = *s_ptr.get().add(row_start + i);
                    let g_val = *g_ptr.get().add(row_start + i);
                    *out_ptr.get().add(row_start + i) = s_val * (g_val - dot);
                }
            }
        });

        vec![Some(out)]
    }
}
