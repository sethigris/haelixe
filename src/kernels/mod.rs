use crate::Tensor;
use rayon::prelude::*;

pub mod activations;
pub mod binary;
pub mod concat;
pub mod layer_norm;
pub mod matmul;
pub mod reduce;
pub mod rope;
pub use rope::{rope_backward, rope_forward};
pub mod loss;
pub use loss::{mse_loss_backward, mse_loss_forward};
pub mod rms_norm;
pub use activations::{
    gelu, gelu_backward, relu, relu_backward, scalar_mul, softmax, softmax_backward,
};
pub use binary::add;
pub use concat::cat_2d;
pub use layer_norm::{layer_norm_backward, layer_norm_forward};
pub use matmul::matmul;
pub use reduce::{sum_all, sum_axis};
pub use rms_norm::{rms_norm_backward, rms_norm_forward};

/// A recursive helper to iterate over an N-dimensional shape,
/// respecting arbitrary strides. This avoids allocating coordinate arrays.
fn strided_loop<F: FnMut(usize)>(
    dim: usize,
    shape: &[usize],
    strides: &[isize],
    current_offset: usize,
    f: &mut F,
) {
    if dim == shape.len() {
        f(current_offset);
        return;
    }

    let stride = strides[dim] as usize;
    for i in 0..shape[dim] {
        strided_loop(dim + 1, shape, strides, current_offset + i * stride, f);
    }
}

/// Fills a tensor with a specific value.
/// Notice how we handle raw pointers: we cast the byte pointer to a typed pointer,
/// allowing us to use standard pointer arithmetic (`add`) safely.
pub fn fill<T: bytemuck::Pod>(tensor: &Tensor, value: T) {
    let shape = tensor.shape.dims();
    let strides = tensor.strides.steps();

    // Get the raw mutable byte pointer, shift it to our view's offset,
    // and cast it to the correct typed pointer.
    let base_byte_ptr = unsafe { tensor.storage.as_mut_ptr().add(tensor.byte_offset) };
    let typed_ptr = base_byte_ptr as *mut T;

    let mut write_val = |offset: usize| {
        unsafe {
            // Because `typed_ptr` is `*mut T`, `.add(offset)` correctly advances
            // the pointer by `offset * size_of::<T>()` bytes.
            typed_ptr.add(offset).write(value);
        }
    };

    strided_loop(0, shape, strides, 0, &mut write_val);
}

/// Copies data from a potentially non-contiguous tensor into a strictly contiguous output tensor.
pub fn copy(from: &Tensor, to: &Tensor) {
    assert_eq!(from.shape, to.shape);
    let num_elements = from.shape.num_elements();

    let in_shape = from.shape.dims();
    let in_strides = from.strides.steps();
    let in_base = from.byte_offset / std::mem::size_of::<f32>();
    let in_ptr = crate::kernels::binary::SyncPtr(from.storage.as_ptr() as *const f32);

    let out_ptr = crate::kernels::binary::SyncMutPtr(to.storage.as_mut_ptr() as *mut f32);

    (0..num_elements).into_par_iter().for_each(|i| {
        let mut offset = 0isize;
        let mut idx = i;
        for d in (0..in_shape.len()).rev() {
            let dim_size = in_shape[d];
            let coord = idx % dim_size;
            idx /= dim_size;
            offset += coord as isize * in_strides[d];
        }
        unsafe {
            *out_ptr.get().add(i) = *in_ptr.get().add(offset as usize + in_base);
        }
    });
}
