use crate::{DType, Tensor};
use rayon::prelude::*;

// --- Thread-Safe Pointer Wrappers for Rayon ---
pub struct SyncPtr<T>(pub *const T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    pub fn get(&self) -> *const T {
        self.0
    }
}

pub struct SyncMutPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }
}

// --- ReLU ---
pub fn relu(tensor: &Tensor) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Auto-download if on GPU
    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    match tensor.dtype {
        DType::F32 => relu_typed::<f32>(&tensor, &out, num_elements),
        DType::F64 => relu_typed::<f64>(&tensor, &out, num_elements),
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

pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let grad_output = grad_output.ensure_cpu(); // Auto-download
    let input = input.ensure_cpu(); // Auto-download
    let out = Tensor::empty(grad_output.dtype, grad_output.shape.clone());
    let num_elements = grad_output.shape.num_elements();

    match grad_output.dtype {
        DType::F32 => relu_backward_typed::<f32>(&grad_output, &input, &out, num_elements),
        DType::F64 => relu_backward_typed::<f64>(&grad_output, &input, &out, num_elements),
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

// --- Scalar Multiplication ---
pub fn scalar_mul(tensor: &Tensor, scalar: f32) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Auto-download
    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        *out_ptr.get().add(i) = *in_ptr.get().add(i) * scalar;
    });
    out
}

// --- Softmax ---
pub fn softmax(tensor: &Tensor) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Auto-download
    assert_eq!(
        tensor.dtype,
        DType::F32,
        "Softmax currently only supports F32"
    );

    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    let last_dim = tensor.shape.dims().last().copied().unwrap_or(1);
    let num_rows = num_elements / last_dim;

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_rows).into_par_iter().for_each(|row_idx| {
        let row_start = row_idx * last_dim;

        // 1. Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..last_dim {
            let val = unsafe { *in_ptr.get().add(row_start + i) };
            if val > max_val {
                max_val = val;
            }
        }

        // 2. Exponentiate and sum
        let mut sum_exp = 0.0f32;
        for i in 0..last_dim {
            let val = unsafe { *in_ptr.get().add(row_start + i) };
            let exp_val = (val - max_val).exp();
            unsafe {
                *out_ptr.get().add(row_start + i) = exp_val;
            }
            sum_exp += exp_val;
        }

        // 3. Normalize
        for i in 0..last_dim {
            unsafe {
                let exp_val = *out_ptr.get().add(row_start + i);
                *out_ptr.get().add(row_start + i) = exp_val / sum_exp;
            }
        }
    });

    out
}

pub fn softmax_backward(dy: &Tensor, y: &Tensor) -> Tensor {
    let dy = dy.ensure_cpu(); // Auto-download
    let y = y.ensure_cpu(); // Auto-download
    let dx = Tensor::empty(dy.dtype, dy.shape.clone());

    let shape = dy.shape.dims();
    let n = *shape.last().unwrap();
    let num_rows = dy.shape.num_elements() / n;

    let dy_ptr = SyncPtr(dy.storage.as_ptr() as *const f32);
    let y_ptr = SyncPtr(y.storage.as_ptr() as *const f32);
    let dx_ptr = SyncMutPtr(dx.storage.as_mut_ptr() as *mut f32);

    (0..num_rows).into_par_iter().for_each(|row| {
        let offset = row * n;
        let mut dot = 0.0f32;

        // 1. Compute dot product of dy and y for this row
        for i in 0..n {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let y_val = unsafe { *y_ptr.get().add(offset + i) };
            dot += dy_val * y_val;
        }

        // 2. Apply the Softmax Jacobian: dx = y * (dy - dot)
        for i in 0..n {
            let dy_val = unsafe { *dy_ptr.get().add(offset + i) };
            let y_val = unsafe { *y_ptr.get().add(offset + i) };
            unsafe {
                *dx_ptr.get().add(offset + i) = y_val * (dy_val - dot);
            }
        }
    });
    dx
}
