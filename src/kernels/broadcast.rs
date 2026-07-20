use crate::{DType, Shape, Tensor};
use rayon::prelude::*;
use std::sync::Mutex;

// Helper to calculate contiguous strides directly from a shape slice
fn get_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

pub fn compute_broadcast(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let ndim = std::cmp::max(shape_a.len(), shape_b.len());
    let mut out_shape = vec![0; ndim];
    let mut strides_a = vec![0; ndim];
    let mut strides_b = vec![0; ndim];

    let pad_a = ndim - shape_a.len();
    let pad_b = ndim - shape_b.len();

    let orig_a = get_contiguous_strides(shape_a);
    let orig_b = get_contiguous_strides(shape_b);

    for i in (0..ndim).rev() {
        let dim_a = if i >= pad_a { shape_a[i - pad_a] } else { 1 };
        let dim_b = if i >= pad_b { shape_b[i - pad_b] } else { 1 };

        if dim_a == dim_b {
            out_shape[i] = dim_a;
            strides_a[i] = if i >= pad_a { orig_a[i - pad_a] } else { 0 };
            strides_b[i] = if i >= pad_b { orig_b[i - pad_b] } else { 0 };
        } else if dim_a == 1 {
            out_shape[i] = dim_b;
            strides_a[i] = 0;
            strides_b[i] = if i >= pad_b { orig_b[i - pad_b] } else { 0 };
        } else if dim_b == 1 {
            out_shape[i] = dim_a;
            strides_b[i] = 0;
            strides_a[i] = if i >= pad_a { orig_a[i - pad_a] } else { 0 };
        } else {
            return None;
        }
    }
    Some((out_shape, strides_a, strides_b))
}

pub fn forward_cpu(
    a: &Tensor,
    b: &Tensor,
    op: u32,
    out_shape: &[usize],
    sa: &[usize],
    sb: &[usize],
) -> Tensor {
    let a_cpu = a.ensure_cpu();
    let b_cpu = b.ensure_cpu();
    let total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; total];
    let ndim = out_shape.len();
    let ap = a_cpu.storage.as_ptr() as *const f32 as usize;
    let bp = b_cpu.storage.as_ptr() as *const f32 as usize;
    let op_ptr = out.as_mut_ptr() as usize;

    (0..total).into_par_iter().for_each(|idx| unsafe {
        let mut rem = idx;
        let mut oa = 0;
        let mut ob = 0;
        for d in (0..ndim).rev() {
            let c = rem % out_shape[d];
            rem /= out_shape[d];
            oa += c * sa[d];
            ob += c * sb[d];
        }
        let va = *((ap as *const f32).add(oa));
        let vb = *((bp as *const f32).add(ob));
        let res = match op {
            0 => va + vb,
            1 => va * vb,
            2 => va - vb,
            3 => va / vb,
            _ => 0.0,
        };
        *((op_ptr as *mut f32).add(idx)) = res;
    });
    Tensor::from_slice(DType::F32, Shape::new(out_shape.to_vec()), &out)
}

pub fn backward_cpu(
    g: &Tensor,
    a: &Tensor,
    b: &Tensor,
    op: u32,
    out_shape: &[usize],
    sa: &[usize],
    sb: &[usize],
) -> (Tensor, Tensor) {
    let g_cpu = g.ensure_cpu();
    let a_cpu = a.ensure_cpu();
    let b_cpu = b.ensure_cpu();
    let total: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let da_m = Mutex::new(vec![0.0f32; a_cpu.shape.num_elements()]);
    let db_m = Mutex::new(vec![0.0f32; b_cpu.shape.num_elements()]);
    let gp = g_cpu.storage.as_ptr() as *const f32 as usize;
    let ap = a_cpu.storage.as_ptr() as *const f32 as usize;
    let bp = b_cpu.storage.as_ptr() as *const f32 as usize;

    // Use the helper instead of Strides.dims()
    let as_orig = get_contiguous_strides(a_cpu.shape.dims());
    let bs_orig = get_contiguous_strides(b_cpu.shape.dims());

    (0..total).into_par_iter().for_each(|idx| unsafe {
        let mut rem = idx;
        let mut oa = 0;
        let mut ob = 0;
        let mut oa_orig = 0;
        let mut ob_orig = 0;
        for d in (0..ndim).rev() {
            let c = rem % out_shape[d];
            rem /= out_shape[d];
            oa += c * sa[d];
            ob += c * sb[d];
            let a_dim = if d >= ndim - a_cpu.shape.dims().len() {
                a_cpu.shape.dims()[d - (ndim - a_cpu.shape.dims().len())]
            } else {
                1
            };
            if a_dim > 1 {
                oa_orig += c * as_orig[d - (ndim - a_cpu.shape.dims().len())];
            }
            let b_dim = if d >= ndim - b_cpu.shape.dims().len() {
                b_cpu.shape.dims()[d - (ndim - b_cpu.shape.dims().len())]
            } else {
                1
            };
            if b_dim > 1 {
                ob_orig += c * bs_orig[d - (ndim - b_cpu.shape.dims().len())];
            }
        }
        let gv = *((gp as *const f32).add(idx));
        let va = *((ap as *const f32).add(oa));
        let vb = *((bp as *const f32).add(ob));
        let (da, db) = match op {
            0 => (gv, gv),
            1 => (gv * vb, gv * va),
            2 => (gv, -gv),
            3 => (gv / vb, -gv * va / (vb * vb)),
            _ => (0.0, 0.0),
        };
        da_m.lock().unwrap()[oa_orig] += da;
        db_m.lock().unwrap()[ob_orig] += db;
    });
    (
        Tensor::from_slice(DType::F32, a_cpu.shape.clone(), &da_m.into_inner().unwrap()),
        Tensor::from_slice(DType::F32, b_cpu.shape.clone(), &db_m.into_inner().unwrap()),
    )
}
