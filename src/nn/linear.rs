// --------------------------------------------------------------------------
// Module: nn::linear
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Implements the foundational affine transformation: y = xW + b.
//
// MATHEMATICAL INVARIANTS:
//   - Weights are initialized using the Kaiming Uniform distribution,
//     which preserves the variance of the activations as they flow
//     through deep networks, preventing vanishing/exploding gradients.
//   - Biases are deterministically initialized to zero.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use super::Module;
use crate::{DType, Shape, Tensor};
use rand::Rng;

/// A standard fully-connected linear layer.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Constructs a new Linear layer with Kaiming Uniform initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Kaiming Uniform Bound: sqrt(6 / fan_in)
        let bound = (6.0 / in_features as f32).sqrt();
        let mut rng = rand::thread_rng();

        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 * bound - bound)
            .collect();

        let b_data: Vec<f32> = vec![0.0; out_features];

        let mut weight =
            Tensor::from_slice(DType::F32, Shape::new([in_features, out_features]), &w_data);
        weight.requires_grad = true;

        let mut bias = Tensor::from_slice(DType::F32, Shape::new([out_features]), &b_data);
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
// This layer currently assumes 2D inputs [Batch, Features]. A mature
// implementation must handle arbitrary N-dimensional broadcasting
// (e.g., [Batch, Seq, Features]) by flattening and unflattening the
// tensor views around the core GEMM kernel.
// --------------------------------------------------------------------------
