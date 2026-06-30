use crate::{DType, Device, Shape, Tensor};
use std::sync::Arc;

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_dim: usize) -> Self {
        let weight_data = vec![1.0f32; hidden_dim];
        Self {
            // Explicitly flag the scaling factor as trainable.
            weight: Tensor::from_slice(DType::F32, Shape::new([hidden_dim]), &weight_data)
                .requires_grad_(true),
            eps: 1e-5,
        }
    }

    pub fn to(&mut self, device: Device) {
        self.weight = self.weight.to(device);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_sync = x.to(self.weight.device.clone());
        let out = crate::kernels::rms_norm_forward(&x_sync, &self.weight, self.eps);

        if x_sync.requires_grad {
            let op = Arc::new(crate::ops::rms_norm::RMSNormOp {
                x: x_sync.clone(),
                weight: self.weight.clone(),
                eps: self.eps,
            });
            out.with_node(op, vec![x_sync, self.weight.clone()])
        } else {
            out
        }
    }
}
