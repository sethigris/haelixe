use crate::{DType, Shape, Tensor};
use rayon::prelude::*;

// 64x64 blocks of f32 take ~16KB. Three of these fit comfortably inside
// a standard 48KB L1 cache. This prevents cache thrashing during the inner loops.
const BLOCK_M: usize = 64;
const BLOCK_N: usize = 64;
const BLOCK_K: usize = 64;

// Reusing the thread-safe pointer wrappers from our binary kernels
struct SyncPtr<T>(*const T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    fn get(&self) -> *const T {
        self.0
    }
}

struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.0
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Temporary bridge: auto-download GPU tensors for CPU matmul
    // TODO: Remove when GPU matmul kernel is implemented
    let a = a.ensure_cpu();
    let b = b.ensure_cpu();

    assert_eq!(a.rank(), 2, "Matmul currently only supports 2D tensors");
    assert_eq!(b.rank(), 2, "Matmul currently only supports 2D tensors");

    let m = a.shape.dims()[0];
    let k_a = a.shape.dims()[1];
    let k_b = b.shape.dims()[0];
    let n = b.shape.dims()[1];

    assert_eq!(k_a, k_b, "Inner dimensions must match ({} vs {})", k_a, k_b);
    assert_eq!(a.dtype, b.dtype, "DTypes must match");

    let k = k_a;
    let out_shape = Shape::new([m, n]);

    // We use zeros because we accumulate (+=) into the output buffer
    let out = Tensor::zeros(a.dtype, out_shape);

    match a.dtype {
        DType::F32 => matmul_typed::<f32>(&a, &b, &out, m, k, n),
        DType::F64 => matmul_typed::<f64>(&a, &b, &out, m, k, n),
        _ => panic!("Unsupported dtype for matmul"),
    }

    out
}
fn matmul_typed<T: bytemuck::Pod + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy>(
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let a_offset = a.byte_offset / std::mem::size_of::<T>();
    let b_offset = b.byte_offset / std::mem::size_of::<T>();
    let out_offset = out.byte_offset / std::mem::size_of::<T>();

    let a_slice = unsafe {
        let slice = a.storage.as_ptr() as *const T;
        std::slice::from_raw_parts(slice.add(a_offset), a.shape.num_elements())
    };
    let b_slice = unsafe {
        let slice = b.storage.as_ptr() as *const T;
        std::slice::from_raw_parts(slice.add(b_offset), b.shape.num_elements())
    };
    let out_slice = unsafe {
        let slice = out.storage.as_mut_ptr() as *mut T;
        std::slice::from_raw_parts_mut(slice.add(out_offset), out.shape.num_elements())
    };

    let a_ptr = SyncPtr(a_slice.as_ptr());
    let b_ptr = SyncPtr(b_slice.as_ptr());
    let out_ptr = SyncMutPtr(out_slice.as_mut_ptr());

    // … rest of the parallel block loop remains the same, using a_ptr.get(), etc.
}
