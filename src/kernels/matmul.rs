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
    // Compute element offsets from byte offsets
    let a_offset_elems = a.byte_offset / std::mem::size_of::<T>();
    let b_offset_elems = b.byte_offset / std::mem::size_of::<T>();
    let out_offset_elems = out.byte_offset / std::mem::size_of::<T>();

    let a_ptr = SyncPtr(unsafe { (a.storage.as_ptr() as *const T).add(a_offset_elems) });
    let b_ptr = SyncPtr(unsafe { (b.storage.as_ptr() as *const T).add(b_offset_elems) });
    let out_ptr = SyncMutPtr(unsafe { (out.storage.as_mut_ptr() as *mut T).add(out_offset_elems) });

    let num_blocks_m = (m + BLOCK_M - 1) / BLOCK_M;
    let num_blocks_n = (n + BLOCK_N - 1) / BLOCK_N;

    (0..num_blocks_m).into_par_iter().for_each(|bm| {
        for bn in 0..num_blocks_n {
            let m_start = bm * BLOCK_M;
            let m_end = (m_start + BLOCK_M).min(m);
            let n_start = bn * BLOCK_N;
            let n_end = (n_start + BLOCK_N).min(n);
            let num_blocks_k = (k + BLOCK_K - 1) / BLOCK_K;

            for bk in 0..num_blocks_k {
                let k_start = bk * BLOCK_K;
                let k_end = (k_start + BLOCK_K).min(k);

                for i in m_start..m_end {
                    let c_row_ptr = unsafe { out_ptr.get().add(i * n) };
                    let a_row_ptr = unsafe { a_ptr.get().add(i * k) };

                    for p in k_start..k_end {
                        let a_val = unsafe { *a_row_ptr.add(p) };
                        let b_row_ptr = unsafe { b_ptr.get().add(p * n) };

                        for j in n_start..n_end {
                            unsafe {
                                let c_ptr = c_row_ptr.add(j);
                                *c_ptr = *c_ptr + (a_val * *b_row_ptr.add(j));
                            }
                        }
                    }
                }
            }
        }
    });
}
