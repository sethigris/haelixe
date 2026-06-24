use crate::{DType, Tensor};
use rayon::prelude::*;

// (Include SyncPtr and SyncMutPtr definitions here, same as binary.rs)
pub struct SyncPtr<T>(pub *const T); // <--- Added pub
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    pub fn get(&self) -> *const T {
        self.0
    } // <--- Added pub
}

pub struct SyncMutPtr<T>(pub *mut T); // <--- Added pub
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    } // <--- Added pub
}

pub fn relu(tensor: &Tensor) -> Tensor {
    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    match tensor.dtype {
        DType::F32 => relu_typed::<f32>(tensor, &out, num_elements),
        DType::F64 => relu_typed::<f64>(tensor, &out, num_elements),
        _ => panic!("Unsupported dtype for relu"),
    }
    out
}

fn relu_typed<T: bytemuck::Pod + PartialOrd + Default + Copy>(
    tensor: &Tensor,
    out: &Tensor,
    num_elements: usize,
) {
    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        let val = *in_ptr.get().add(i);
        let zero = T::default();
        *out_ptr.get().add(i) = if val > zero { val } else { zero };
    });
}

/// ReLU backward is just passing the gradient through if input > 0, else 0.
pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let out = Tensor::empty(grad_output.dtype, grad_output.shape.clone());
    let num_elements = grad_output.shape.num_elements();

    match grad_output.dtype {
        DType::F32 => relu_backward_typed::<f32>(grad_output, input, &out, num_elements),
        DType::F64 => relu_backward_typed::<f64>(grad_output, input, &out, num_elements),
        _ => panic!("Unsupported dtype for relu_backward"),
    }
    out
}

fn relu_backward_typed<T: bytemuck::Pod + PartialOrd + Default + Copy>(
    grad: &Tensor,
    input: &Tensor,
    out: &Tensor,
    num_elements: usize,
) {
    let grad_ptr = SyncPtr(grad.storage.as_ptr() as *const T);
    let in_ptr = SyncPtr(input.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        let g = *grad_ptr.get().add(i);
        let x = *in_ptr.get().add(i);
        let zero = T::default();
        *out_ptr.get().add(i) = if x > zero { g } else { zero };
    });
}

/// Multiplies every element in the tensor by a single scalar value.
/// Used heavily in Attention for scaling: Q @ K^T * (1 / sqrt(d_k))
pub fn scalar_mul(tensor: &Tensor, scalar: f32) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Bridge for now
    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        *out_ptr.get().add(i) = *in_ptr.get().add(i) * scalar;
    });
    out
}

/// Numerically stable Softmax applied along the LAST dimension.
/// Transforms raw scores into probabilities where each row sums to 1.0.
pub fn softmax(tensor: &Tensor) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Bridge for now
    assert_eq!(
        tensor.dtype,
        DType::F32,
        "Softmax currently only supports F32"
    );

    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    // Softmax is applied along the last dimension (the "row").
    let last_dim = tensor.shape.dims().last().copied().unwrap_or(1);
    let num_rows = num_elements / last_dim;

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    // Parallelize over the rows. Each thread handles one complete row.
    (0..num_rows).into_par_iter().for_each(|row_idx| {
        let row_start = row_idx * last_dim;

        // 1. Find the maximum value in the row (Numerical stability trick)
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..last_dim {
            let val = unsafe { *in_ptr.get().add(row_start + i) };
            if val > max_val {
                max_val = val;
            }
        }

        // 2. Exponentiate (val - max) and calculate the sum of the row
        let mut sum_exp = 0.0f32;
        for i in 0..last_dim {
            let val = unsafe { *in_ptr.get().add(row_start + i) };
            let exp_val = (val - max_val).exp();
            unsafe {
                *out_ptr.get().add(row_start + i) = exp_val;
            }
            sum_exp += exp_val;
        }

        // 3. Normalize by the sum so the row adds up to exactly 1.0
        for i in 0..last_dim {
            unsafe {
                let exp_val = *out_ptr.get().add(row_start + i);
                *out_ptr.get().add(row_start + i) = exp_val / sum_exp;
            }
        }
    });

    out
}
