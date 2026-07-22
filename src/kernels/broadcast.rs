use crate::{DType, Shape, Tensor};

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

    // Safe slices – byte_offset is 0 for the contiguous tensors we operate on
    let a_data = unsafe { a_cpu.storage.as_f32_slice() };
    let b_data = unsafe { b_cpu.storage.as_f32_slice() };

    let a_data = unsafe { a_cpu.storage.as_f32_slice() };
    let b_data = unsafe { b_cpu.storage.as_f32_slice() };
    eprintln!("a_data first 6: {:?}", &a_data[..a_data.len().min(6)]);
    eprintln!("b_data first 3: {:?}", &b_data[..b_data.len().min(3)]);

    for idx in 0..total {
        let mut rem = idx;
        let mut oa = 0;
        let mut ob = 0;
        for d in (0..ndim).rev() {
            let c = rem % out_shape[d];
            rem /= out_shape[d];
            oa += c * sa[d];
            ob += c * sb[d];
        }
        let va = a_data[oa];
        let vb = b_data[ob];
        out[idx] = match op {
            0 => va + vb,
            1 => va * vb,
            2 => va - vb,
            3 => va / vb,
            _ => 0.0,
        };
    }
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
    let total = out_shape.iter().product();
    let ndim = out_shape.len();
    let a_elems = a_cpu.shape.num_elements();
    let b_elems = b_cpu.shape.num_elements();
    let mut da = vec![0.0f32; a_elems];
    let mut db = vec![0.0f32; b_elems];

    let g_data = unsafe { g_cpu.storage.as_f32_slice() };
    let a_data = unsafe { a_cpu.storage.as_f32_slice() };
    let b_data = unsafe { b_cpu.storage.as_f32_slice() };

    let as_orig = get_contiguous_strides(a_cpu.shape.dims());
    let bs_orig = get_contiguous_strides(b_cpu.shape.dims());

    for idx in 0..total {
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
        let gv = g_data[idx];
        let va = a_data[oa];
        let vb = b_data[ob];
        let (da_val, db_val) = match op {
            0 => (gv, gv),
            1 => (gv * vb, gv * va),
            2 => (gv, -gv),
            3 => (gv / vb, -gv * va / (vb * vb)),
            _ => (0.0, 0.0),
        };
        da[oa_orig] += da_val;
        db[ob_orig] += db_val;
    }
    (
        Tensor::from_slice(DType::F32, a_cpu.shape.clone(), &da),
        Tensor::from_slice(DType::F32, b_cpu.shape.clone(), &db),
    )
}
