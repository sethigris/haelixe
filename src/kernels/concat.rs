use crate::{Shape, Tensor};

pub fn cat_2d(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cat_2d requires at least one tensor");
    let cols = tensors[0].shape.dims()[1];
    let total_rows: usize = tensors.iter().map(|t| t.shape.dims()[0]).sum();
    let out = Tensor::zeros(tensors[0].dtype, Shape::new([total_rows, cols]));
    let out_slice = unsafe {
        let slice = out.storage.as_f32_slice_mut();
        &mut slice[..total_rows * cols]
    };
    let mut row_offset = 0;
    for t in tensors {
        let t_cpu = t.ensure_cpu();
        let t_rows = t_cpu.shape.dims()[0];
        let t_slice = unsafe {
            let slice = t_cpu.storage.as_f32_slice();
            &slice[..t_rows * cols]
        };
        out_slice[row_offset * cols..(row_offset + t_rows) * cols].copy_from_slice(t_slice);
        row_offset += t_rows;
    }
    out
}
