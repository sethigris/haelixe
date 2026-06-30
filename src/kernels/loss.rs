use crate::Tensor;
use crate::kernels::activations::{SyncMutPtr, SyncPtr};
use rayon::prelude::*;

/// Computes the Mean Squared Error: (1/N) * sum((pred - target)^2)
pub fn mse_loss_forward(pred: &Tensor, target: &Tensor) -> f32 {
    let pred = pred.ensure_cpu();
    let target = target.ensure_cpu();
    let n = pred.shape.num_elements();

    let p_ptr = SyncPtr(pred.storage.as_ptr() as *const f32);
    let t_ptr = SyncPtr(target.storage.as_ptr() as *const f32);

    let squared_errors: f32 = (0..n)
        .into_par_iter()
        .map(|i| unsafe {
            let diff = *p_ptr.get().add(i) - *t_ptr.get().add(i);
            diff * diff
        })
        .sum();

    squared_errors / n as f32
}

/// Computes the gradient of MSE: (2/N) * (pred - target)
pub fn mse_loss_backward(pred: &Tensor, target: &Tensor) -> Tensor {
    let pred = pred.ensure_cpu();
    let target = target.ensure_cpu();
    let n = pred.shape.num_elements();
    let grad = Tensor::empty(pred.dtype, pred.shape.clone());

    let p_ptr = SyncPtr(pred.storage.as_ptr() as *const f32);
    let t_ptr = SyncPtr(target.storage.as_ptr() as *const f32);
    let g_ptr = SyncMutPtr(grad.storage.as_mut_ptr() as *mut f32);

    let scale = 2.0 / n as f32;

    (0..n).into_par_iter().for_each(|i| unsafe {
        let diff = *p_ptr.get().add(i) - *t_ptr.get().add(i);
        *g_ptr.get().add(i) = diff * scale;
    });

    grad
}
