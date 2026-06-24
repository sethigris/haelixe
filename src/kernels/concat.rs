use crate::{Shape, Tensor};

/// Concatenates a list of 2D tensors along a new leading dimension (dim 0).
pub fn cat_2d(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "Cannot concatenate empty tensor list");
    let n = tensors.len();
    let s = tensors[0].shape.dims()[0];
    let d = tensors[0].shape.dims()[1];

    let out = Tensor::empty(tensors[0].dtype, Shape::new([n, s, d]));

    // Physically copy each 2D slice into the 3D buffer
    for (i, t) in tensors.iter().enumerate() {
        let out_slice = out.get_2d_slice(i);
        crate::kernels::copy(t, &out_slice);
    }
    out
}
