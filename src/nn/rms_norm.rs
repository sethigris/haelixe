// --------------------------------------------------------------------------
// Module: nn::rms_norm
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Implements Root Mean Square Layer Normalization (RMSNorm).
//
// HISTORICAL CONTEXT:
//   Upgraded during the 100-Hour Hardening Phase (Pillar 3) to support
//   Device-Aware Initialization.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use crate::{Tensor, DType, Shape, Device};
use super::Module;

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_dim: usize, device: &Device) -> Self {
        let ones = vec![1.0f32; hidden_dim];
        let mut weight = Tensor::from_slice(
            DType::F32, 
            Shape::new([hidden_dim]), 
            &ones
        ).to(device.clone());
        weight.requires_grad = true;
        
        Self { weight, eps: 1e-5 }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let out = crate::kernels::rms_norm::rms_norm_forward(
            x, &self.weight, self.eps
        );
        
        if x.requires_grad || self.weight.requires_grad {
            let op = std::sync::Arc::new(crate::ops::rms_norm::RMSNormOp {
                x: x.clone(),
                weight: self.weight.clone(),
                eps: self.eps,
            });
            out.with_node(op, vec![x.clone(), self.weight.clone()])
        } else {
            out
        }
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}
