use crate::Tensor;
use crate::kernels::activations::{SyncMutPtr, SyncPtr};
use rayon::prelude::*;
use std::sync::Mutex;

pub fn rms_norm_forward(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let x = x.ensure_cpu();
    let weight = weight.ensure_cpu();
    let out = Tensor::empty(x.dtype, x.shape.clone());

    let shape = x.shape.dims();
    let last_dim = *shape.last().unwrap();
    let num_rows = x.shape.num_elements() / last_dim;

    let x_ptr = SyncPtr(x.storage.as_ptr() as *const f32);
    let w_ptr = SyncPtr(weight.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_rows).into_par_iter().for_each(|row| {
        let offset = row * last_dim;
        let mut sum_sq = 0.0f32;
        for i in 0..last_dim {
            let val = unsafe { *x_ptr.get().add(offset + i) };
            sum_sq += val * val;
        }
        let rms = (sum_sq / last_dim as f32 + eps).sqrt();

        for i in 0..last_dim {
            let val = unsafe { *x_ptr.get().add(offset + i) };
            let w = unsafe { *w_ptr.get().add(i) };
            unsafe {
                *out_ptr.get().add(offset + i) = (val / rms) * w;
            }
        }
    });
    out
}

pub fn rms_norm_backward(dy: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> (Tensor, Tensor) {
    let dy = dy.ensure_cpu();
    let x = x.ensure_cpu();
    let weight = weight.ensure_cpu();

    let dx = Tensor::empty(x.dtype, x.shape.clone());
    let dw = Tensor::empty(weight.dtype, weight.shape.clone());

    let shape = x.shape.dims();
    let last_dim = *shape.last().unwrap();
    let num_rows = x.shape.num_elements() / last_dim;

    let dy_ptr = SyncPtr(dy.storage.as_ptr() as *const f32);
    let x_ptr = SyncPtr(x.storage.as_ptr() as *const f32);
    let w_ptr = SyncPtr(weight.storage.as_ptr() as *const f32);
    let dx_ptr = SyncMutPtr(dx.storage.as_mut_ptr() as *mut f32);

    let dw_accum = Mutex::new(vec![0.0f32; last_dim]);

    (0..num_rows).into_par_iter().for_each(|row| {
        let offset = row * last_dim;
        let mut sum_sq = 0.0f32;
        for i in 0..last_dim {
            let val = unsafe { *x_ptr.get().add(offset + i) };
            sum_sq += val * val;
        }
        let rms = (sum_sq / last_dim as f32 + eps).sqrt();
        let r3 = rms * rms * rms;

        let mut sum_dy_x_w = 0.0f32;
        for i in 0..last_dim {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let w_val = unsafe { *w_ptr.get().add(i) };
            sum_dy_x_w += dy_val * x_val * w_val;
        }

        let mut local_dw = vec![0.0f32; last_dim];

        for i in 0..last_dim {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let w_val = unsafe { *w_ptr.get().add(i) };

            // Mathematically rigorous RMSNorm gradient
            let dx_val = (dy_val * w_val) / rms - (x_val * sum_dy_x_w) / (last_dim as f32 * r3);
            unsafe {
                *dx_ptr.get().add(offset + i) = dx_val;
            }

            local_dw[i] = dy_val * (x_val / rms);
        }

        let mut global_dw = dw_accum.lock().unwrap();
        for i in 0..last_dim {
            global_dw[i] += local_dw[i];
        }
    });

    let dw_ptr = SyncMutPtr(dw.storage.as_mut_ptr() as *mut f32);
    let global_dw = dw_accum.lock().unwrap();
    for i in 0..last_dim {
        unsafe {
            *dw_ptr.get().add(i) = global_dw[i];
        }
    }

    (dx, dw)
}
