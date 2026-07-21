use rayon::prelude::*;
use std::sync::Mutex;
use crate::{DType, Device, Shape, Tensor};

// --------------------------------------------------------------------------
// Module: kernels::reduce
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Orchestrates tensor reduction operations (Sum, Mean) across both CPU
//   and GPU backends. For GPU, it dispatches a Hybrid SRAM Tree-Reduction
//   WGSL shader. For CPU, or for axis-specific reductions, it utilizes
//   parallelized Rayon iterators.
//
// HISTORICAL CONTEXT:
//   Forged during the 100-Hour Hardening Phase to close the gap between
//   toy autodiff engines and production frameworks. Without reductions,
//   Loss functions (MSE, Cross-Entropy) and Normalization layers (LayerNorm)
//   cannot be executed natively. The GPU implementation specifically
//   bypasses WebGPU's lack of native `atomic<f32>` by reducing to
//   workgroup sums in L1 SRAM, then finalizing on the CPU.
//
// INVARIANTS:
//   - `op_code` must be strictly 0 (Sum) or 1 (Mean).
//   - `axis` in `sum_axis` must be strictly less than the tensor's `ndim`.
//
// FAILURE MODES:
//   - Axis Out of Bounds: `sum_axis` will panic if the requested axis
//     exceeds the tensor's dimensionality. This is a fatal programmer
//     error, not a recoverable runtime condition.
//   - GPU Readback Failure: If the WebGPU staging buffer fails to map
//     during the hybrid CPU finish, the thread will panic on unwrap.
//
// CALL GRAPH:
//   Called by: `Tensor::sum()`, `Tensor::mean()`, and backward passes of
//              broadcasting operations (e.g., `BinaryBroadcastOp`).
//   Calls: `GpuContext::reduce_gpu()`, `Tensor::from_slice()`.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Reviewed by: [Pending Peer Review]
//   Date: 2026-07-21
// --------------------------------------------------------------------------

/// Executes the forward pass for global reduction (Sum or Mean).
/// Routes to the GPU WGSL shader if the tensor resides in VRAM,
/// otherwise falls back to a highly parallelized CPU Rayon iterator.
pub fn reduce_forward(x: &Tensor, op_code: u32) -> Tensor {
    let total = x.shape.num_elements();

    // Fallback to CPU if not on GPU
    if !x.device.is_gpu() {
        let x_cpu = x.ensure_cpu();
        let slice = unsafe {
            std::slice::from_raw_parts(x_cpu.storage.as_ptr() as *const f32, total)
        };
        let sum: f32 = slice.iter().sum();
        let val = if op_code == 1 {
            sum / total as f32
        } else {
            sum
        };
        return Tensor::from_slice(DType::F32, Shape::new([1]), &[val]);
    }

    let ctx = match &x.device {
        Device::Gpu(c) => c.clone(),
        _ => unreachable!(),
    };
    crate::gpu::GpuContext::reduce_gpu(&ctx, x, total, op_code)
}

/// Computes the gradient for a global reduction operation.
/// The gradient of `sum()` is broadcasting 1.0 * grad to the original shape.
/// The gradient of `mean()` is broadcasting (1/N) * grad to the original shape.
pub fn reduce_backward(grad: &Tensor, orig_shape: &Shape, op_code: u32) -> Tensor {
    let total = orig_shape.num_elements();
    let scale = if op_code == 1 {
        1.0 / total as f32
    } else {
        1.0
    };

    // Create a tensor of 1.0s with the original shape on the same device.
    // This acts as the carrier for the broadcasted gradient.
    let ones_data = vec![1.0f32; total];
    let ones = Tensor::from_slice(DType::F32, orig_shape.clone(), &ones_data)
        .to(grad.device.clone());

    // Scale the incoming scalar gradient (e.g., multiply by 1/N for mean)
    let grad_scaled = grad.mul_scalar(scale);

    // Broadcast the scalar gradient to the original N-dimensional shape
    // using our Zero-Copy Broadcasting engine (op_code 1 = Mul).
    ones.binary_broadcast(&grad_scaled, 1)
}

/// Reduces a tensor along a specific axis, dropping that dimension.
/// e.g., Shape([4, 2]) summed along axis 0 becomes Shape([2]).
/// This is critical for gradient accumulation in broadcasting backward passes.
pub fn sum_axis(tensor: &Tensor, axis: usize) -> Tensor {
    let t = tensor.ensure_cpu();
    let dims = t.shape.dims();
    let ndim = dims.len();
    assert!(
        axis < ndim,
        "Axis {} out of bounds for {}D tensor",
        axis,
        ndim
    );

    let mut out_shape: Vec<usize> = dims.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let total_in = t.shape.num_elements();
    let total_out: usize = out_shape.iter().product();

    // Calculate contiguous strides for the output tensor to map N-dim 
    // coordinates back to a flat 1D memory offset efficiently.
    let mut out_strides = vec![0usize; out_shape.len()];
    if !out_shape.is_empty() {
        let mut acc = 1;
        for i in (0..out_shape.len()).rev() {
            out_strides[i] = acc;
            acc *= out_shape[i];
        }
    }

    // We wrap the output vector in a Mutex to guarantee mathematical
    // correctness during parallel accumulation. While a Mutex lock per
    // element introduces contention, it strictly prevents the catastrophic
    // read-modify-write race conditions inherent in raw pointer arithmetic.
    let out_mutex = Mutex::new(vec![0.0f32; total_out]);
    let t_ptr = t.storage.as_ptr() as *const f32 as usize;

    (0..total_in).into_par_iter().for_each(|idx| unsafe {
        // Decompose flat index into N-dim coordinates
        let mut rem = idx;
        let mut coords = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            coords[d] = rem % dims[d];
            rem /= dims[d];
        }

        // Map the N-dim coordinates to the reduced output flat index
        let mut out_flat = 0;
        let mut out_dim_idx = 0;
        for d in 0..ndim {
            if d != axis {
                out_flat += coords[d] * out_strides[out_dim_idx];
                out_dim_idx += 1;
            }
        }

        let val = *((t_ptr as *const f32).add(idx));
        
        // Thread-safe accumulation
        let mut lock = out_mutex.lock().unwrap();
        lock[out_flat] += val;
    });

    let out = out_mutex.into_inner().unwrap();
    Tensor::from_slice(DType::F32, Shape::new(out_shape), &out)
}

/// Computes the global sum of all elements. 
/// Acts as a compatibility wrapper for legacy ops that expect `sum_all`.
pub fn sum_all(tensor: &Tensor) -> Tensor {
    reduce_forward(tensor, 0)
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
// The `sum_axis` implementation currently relies on a `Mutex` for parallel
// accumulation. While mathematically pure and strictly safe from the race
// conditions that plagued earlier raw-pointer iterations, the lock contention
// will severely bottleneck performance on massive tensors. 
//
// The engineer who inherits this file is urged to replace the `Mutex` with
// a thread-local chunking strategy (e.g., Rayon's `fold` and `reduce`), or
// migrate the axis-reduction logic entirely to a dedicated WGSL compute
// shader that utilizes Workgroup SRAM atomics.
// --------------------------------------------------------------------------