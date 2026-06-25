use crate::kernels::binary::{SyncMutPtr, SyncPtr};
use crate::{DType, Shape, Tensor};
use rayon::prelude::*;

/// Forward pass: Computes output, mean, and reciprocal standard deviation (rstd).
pub fn layer_norm_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> (Tensor, Tensor, Tensor) {
    let x = x.ensure_cpu();
    let weight = weight.ensure_cpu();
    let bias = bias.ensure_cpu();

    let shape = x.shape.dims();
    let n = *shape.last().unwrap(); // Normalized shape (last dim)
    let num_rows = x.shape.num_elements() / n;

    let out = Tensor::empty(DType::F32, x.shape.clone());
    let mean = Tensor::empty(DType::F32, Shape::new([num_rows]));
    let rstd = Tensor::empty(DType::F32, Shape::new([num_rows]));

    let x_ptr = SyncPtr(x.storage.as_ptr() as *const f32);
    let w_ptr = SyncPtr(weight.storage.as_ptr() as *const f32);
    let b_ptr = SyncPtr(bias.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);
    let mean_ptr = SyncMutPtr(mean.storage.as_mut_ptr() as *mut f32);
    let rstd_ptr = SyncMutPtr(rstd.storage.as_mut_ptr() as *mut f32);

    (0..num_rows).into_par_iter().for_each(|row| {
        let offset = row * n;

        // 1. Compute Mean
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += unsafe { *x_ptr.get().add(offset + i) };
        }
        let m = sum / n as f32;
        unsafe {
            *mean_ptr.get().add(row) = m;
        }

        // 2. Compute Variance and Rstd
        let mut var_sum = 0.0f32;
        for i in 0..n {
            let diff = unsafe { *x_ptr.get().add(offset + i) } - m;
            var_sum += diff * diff;
        }
        let r = 1.0 / (var_sum / n as f32 + eps).sqrt();
        unsafe {
            *rstd_ptr.get().add(row) = r;
        }

        // 3. Normalize, Scale, and Shift
        for i in 0..n {
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let w_val = unsafe { *w_ptr.get().add(i) };
            let b_val = unsafe { *b_ptr.get().add(i) };
            let norm = (x_val - m) * r;
            unsafe {
                *out_ptr.get().add(offset + i) = norm * w_val + b_val;
            }
        }
    });

    (out, mean, rstd)
}

/// Backward pass: Computes dx, dweight, and dbias in one optimized sweep.
pub fn layer_norm_backward(
    dy: &Tensor,
    x: &Tensor,
    weight: &Tensor,
    mean: &Tensor,
    rstd: &Tensor,
    n: usize,
) -> (Tensor, Tensor, Tensor) {
    let dy = dy.ensure_cpu();
    let x = x.ensure_cpu();
    let weight = weight.ensure_cpu();
    let mean = mean.ensure_cpu();
    let rstd = rstd.ensure_cpu();

    let num_rows = x.shape.num_elements() / n;
    let dx = Tensor::empty(DType::F32, x.shape.clone());

    // Accumulators for dweight and dbias (reduced across all rows)
    let mut dweight_acc = vec![0.0f32; n];
    let mut dbias_acc = vec![0.0f32; n];

    let dy_ptr = SyncPtr(dy.storage.as_ptr() as *const f32);
    let x_ptr = SyncPtr(x.storage.as_ptr() as *const f32);
    let w_ptr = SyncPtr(weight.storage.as_ptr() as *const f32);
    let m_ptr = SyncPtr(mean.storage.as_ptr() as *const f32);
    let r_ptr = SyncPtr(rstd.storage.as_ptr() as *const f32);
    let dx_ptr = SyncMutPtr(dx.storage.as_mut_ptr() as *mut f32);

    // 1. Compute dx row-by-row in parallel
    (0..num_rows).into_par_iter().for_each(|row| {
        let offset = row * n;
        let m = unsafe { *m_ptr.get().add(row) };
        let r = unsafe { *r_ptr.get().add(row) };

        let mut dy_w_sum = 0.0f32;
        let mut dy_w_x_hat_sum = 0.0f32;

        for i in 0..n {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let w_val = unsafe { *w_ptr.get().add(i) };
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let x_hat = (x_val - m) * r;

            let dy_w = dy_val * w_val;
            dy_w_sum += dy_w;
            dy_w_x_hat_sum += dy_w * x_hat;
        }

        for i in 0..n {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let w_val = unsafe { *w_ptr.get().add(i) };
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let x_hat = (x_val - m) * r;

            let dy_w = dy_val * w_val;
            let dx_val = (r / n as f32) * (n as f32 * dy_w - dy_w_sum - x_hat * dy_w_x_hat_sum);
            unsafe {
                *dx_ptr.get().add(offset + i) = dx_val;
            }
        }
    });

    // 2. Compute dweight and dbias (Sequential reduction over rows is fast enough for N < 1024)
    for row in 0..num_rows {
        let offset = row * n;
        let m = unsafe { *m_ptr.get().add(row) };
        let r = unsafe { *r_ptr.get().add(row) };
        for i in 0..n {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let x_val = unsafe { *x_ptr.get().add(offset + i) };
            let x_hat = (x_val - m) * r;
            dbias_acc[i] += dy_val;
            dweight_acc[i] += dy_val * x_hat;
        }
    }

    let dweight = Tensor::from_slice(DType::F32, Shape::new([n]), &dweight_acc);
    let dbias = Tensor::from_slice(DType::F32, Shape::new([n]), &dbias_acc);

    (dx, dweight, dbias)
}
