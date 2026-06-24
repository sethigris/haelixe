use crate::{DType, Strides, Tensor};
use rayon::prelude::*;

// Raw pointers are not `Send` or `Sync` by default in Rust.
// Since we manually guarantee thread safety (our parallel iterator ensures
// no overlapping mutable accesses), we opt-in.
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

#[inline(always)]
fn map_1d_to_nd_offset(mut index: usize, shape: &[usize], strides: &[isize]) -> usize {
    let mut offset = 0isize;

    for i in (0..shape.len()).rev() {
        let dim_size = shape[i];
        let coord = index % dim_size;
        index /= dim_size;
        offset += coord as isize * strides[i];
    }
    offset as usize
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    // Compute the broadcasted output shape
    let out_shape = a
        .shape
        .broadcast(&b.shape)
        .expect("Shapes are not broadcastable");
    assert_eq!(a.dtype, b.dtype, "DTypes must match for element-wise add");

    let out = Tensor::empty(a.dtype, out_shape.clone());
    let num_elements = out_shape.num_elements();

    // Compute broadcasted strides for both inputs based on the output shape
    let a_strides = Strides::broadcast_to(&a.shape, &a.strides, &out_shape);
    let b_strides = Strides::broadcast_to(&b.shape, &b.strides, &out_shape);

    match a.dtype {
        DType::F32 => add_typed::<f32>(a, &a_strides, b, &b_strides, &out, num_elements),
        DType::F64 => add_typed::<f64>(a, &a_strides, b, &b_strides, &out, num_elements),
        _ => panic!("Unsupported dtype for add kernel"),
    }

    out
}

fn add_typed<T: bytemuck::Pod + std::ops::Add<Output = T>>(
    a: &Tensor,
    a_strides: &Strides,
    b: &Tensor,
    b_strides: &Strides,
    out: &Tensor,
    num_elements: usize,
) {
    // We pass the OUTPUT shape to the mapping function so it correctly
    // handles the broadcasted dimensions.
    let out_shape = out.shape.dims();

    let a_shape = out_shape;
    let a_strides = a_strides.steps();
    let a_base = a.byte_offset / std::mem::size_of::<T>();

    let b_shape = out_shape;
    let b_strides = b_strides.steps();
    let b_base = b.byte_offset / std::mem::size_of::<T>();

    let out_base = out.byte_offset / std::mem::size_of::<T>();

    let a_ptr = SyncPtr(a.storage.as_ptr() as *const T);
    let b_ptr = SyncPtr(b.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);
    (0..num_elements).into_par_iter().for_each(|i| {
        // Map the flat index using the OUTPUT shape, but the BROADCASTED strides
        let a_idx = map_1d_to_nd_offset(i, a_shape, a_strides) + a_base;
        let b_idx = map_1d_to_nd_offset(i, b_shape, b_strides) + b_base;
        let out_idx = i + out_base;

        unsafe {
            let val_a = *a_ptr.get().add(a_idx);
            let val_b = *b_ptr.get().add(b_idx);
            *out_ptr.get().add(out_idx) = val_a + val_b;
        }
    });
}
