use crate::ops::linear_fused::FusedLinearOp;
use crate::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = (1.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| fastrand::f32() * 2.0 * bound - bound)
            .collect();

        let bias_data = vec![0.0f32; out_features];

        let weight = Tensor::from_slice(
            DType::F32,
            Shape::new([in_features, out_features]),
            &weight_data,
        )
        .requires_grad_(true);

        let bias = Tensor::from_slice(DType::F32, Shape::new([out_features]), &bias_data)
            .requires_grad_(true);

        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if x.rank() == 3 {
            // Systems Trick for Transformers: Flatten [Batch, Seq, In] -> [Batch*Seq, In]
            let b = x.shape.dims()[0];
            let s = x.shape.dims()[1];
            let in_f = x.shape.dims()[2];
            let out_f = self.weight.shape.dims()[1];

            let x_flat = x.view(crate::Shape::new([b * s, in_f]));
            let out_flat = x_flat.matmul(&self.weight);
            let out_biased = out_flat.add(&self.bias);

            // Reshape back to [Batch, Seq, Out]
            out_biased.view(crate::Shape::new([b, s, out_f]))
        } else if x.rank() == 2 {
            // Standard 2D forward
            let out = x.matmul(&self.weight);
            out.add(&self.bias)
        } else {
            panic!(
                "Linear layer currently only supports 2D and 3D inputs, got {}D",
                x.rank()
            );
        }
    }
}
