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

/// Forward pass: Computes Cross-Entropy Loss using the LogSumExp trick.
pub fn cross_entropy_forward(
    logits: &[f32],
    targets: &[u32],
    batch_size: usize,
    num_classes: usize,
) -> (f32, Vec<f32>) {
    let mut losses = vec![0.0f32; batch_size];
    let mut softmax_probs = vec![0.0f32; batch_size * num_classes];

    // Zip parallel iterators to safely mutate disjoint chunks
    let logits_chunks = logits.par_chunks(num_classes);
    let targets_iter = targets.par_iter();
    let losses_iter = losses.par_iter_mut();
    let probs_chunks = softmax_probs.par_chunks_mut(num_classes);

    logits_chunks
        .zip(targets_iter)
        .zip(losses_iter)
        .zip(probs_chunks)
        .for_each(|(((row, &target), loss), probs_row)| {
            let target_idx = target as usize;

            // LogSumExp Trick
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_sum_exp = sum_exp.ln() + max_val;

            *loss = -row[target_idx] + log_sum_exp;

            for j in 0..num_classes {
                probs_row[j] = (row[j] - max_val).exp() / sum_exp;
            }
        });

    let total_loss = losses.par_iter().sum::<f32>() / batch_size as f32;
    (total_loss, softmax_probs)
}

/// Backward pass: Gradient is (Softmax - OneHot).
pub fn cross_entropy_backward(
    softmax_probs: &[f32],
    targets: &[u32],
    batch_size: usize,
    num_classes: usize,
) -> Vec<f32> {
    let mut grads = vec![0.0f32; batch_size * num_classes];
    let scale = 1.0 / batch_size as f32;

    let probs_chunks = softmax_probs.par_chunks(num_classes);
    let targets_iter = targets.par_iter();
    let grads_chunks = grads.par_chunks_mut(num_classes);

    probs_chunks.zip(targets_iter).zip(grads_chunks).for_each(
        |((probs_row, &target), grads_row)| {
            let target_idx = target as usize;
            for j in 0..num_classes {
                let mut grad = probs_row[j];
                if j == target_idx {
                    grad -= 1.0;
                }
                grads_row[j] = grad * scale;
            }
        },
    );

    grads
}
