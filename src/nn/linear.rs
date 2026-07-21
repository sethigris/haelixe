// --------------------------------------------------------------------------
// Module: nn::linear
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Implements the foundational affine transformation: y = xW + b.
//
// HISTORICAL CONTEXT:
//   Upgraded during the 100-Hour Hardening Phase (Pillar 3) to support
//   Device-Aware Initialization. Weights are now born directly in VRAM,
//   eliminating the PCIe Initialization Tax that plagues legacy frameworks.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use super::Module;
use crate::{DType, Device, Shape, Tensor};
use rand::Rng;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Constructs a new Linear layer and births it directly on the target device.
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
        let bound = (6.0 / in_features as f32).sqrt();
        let mut rng = rand::thread_rng();

        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 * bound - bound)
            .collect();

        let b_data: Vec<f32> = vec![0.0; out_features];

        // Generate on CPU, instantly stream to Device, drop CPU buffer.
        let mut weight =
            Tensor::from_slice(DType::F32, Shape::new([in_features, out_features]), &w_data)
                .to(device.clone());
        weight.requires_grad = true;

        let mut bias =
            Tensor::from_slice(DType::F32, Shape::new([out_features]), &b_data).to(device.clone());
        bias.requires_grad = true;

        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add(&self.bias)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
// This layer still relies on a temporary CPU `Vec` to generate the random
// noise before streaming it to the GPU. A mature systems engineer will
// eventually replace this with a GPU-side PRNG WGSL shader, allowing the
// weights to be initialized entirely on the silicon without ever touching
// system RAM.
// --------------------------------------------------------------------------
