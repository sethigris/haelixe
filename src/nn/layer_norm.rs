use crate::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct LayerNorm {
    pub weight: Tensor, // gamma
    pub bias: Tensor,   // beta
    pub eps: f32,
    pub normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        let weight = Tensor::ones(DType::F32, Shape::new([normalized_shape])).requires_grad_(true);
        let bias = Tensor::zeros(DType::F32, Shape::new([normalized_shape])).requires_grad_(true);
        Self {
            weight,
            bias,
            eps,
            normalized_shape,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to(self.weight.device.clone());
        let (out, mean, rstd) =
            crate::kernels::layer_norm_forward(&x, &self.weight, &self.bias, self.eps);

        if x.requires_grad || self.weight.requires_grad || self.bias.requires_grad {
            let op = Arc::new(crate::ops::layer_norm::LayerNormOp {
                x: x.clone(),
                weight: self.weight.clone(),
                mean,
                rstd,
                normalized_shape: self.normalized_shape,
            });
            out.with_node(op, vec![x.clone(), self.weight.clone(), self.bias.clone()])
        } else {
            out
        }
    }

    pub fn to(&mut self, device: crate::Device) {
        self.weight = self.weight.to(device.clone());
        self.bias = self.bias.to(device.clone());
    }
}
