use crate::{Device, GpuContext, Shape, Tensor};
use rand::Rng;
use std::sync::Arc;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let fan_in = in_features as f32;

        // Kaiming Uniform Bound for Weights
        let weight_bound = (6.0 / fan_in).sqrt();
        // Uniform Bound for Bias
        let bias_bound = 1.0 / fan_in.sqrt();

        let mut rng = rand::thread_rng();

        // Initialize Weights
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-weight_bound..weight_bound))
            .collect();

        // Initialize Bias
        let bias_data: Vec<f32> = (0..out_features)
            .map(|_| rng.gen_range(-bias_bound..bias_bound))
            .collect();

        Self {
            // Explicitly flag weights as trainable so the Autograd
            // engine registers them in the gradient map for the optimizer.
            weight: Tensor::from_slice(
                crate::DType::F32,
                crate::Shape::new([in_features, out_features]),
                &weight_data,
            )
            .requires_grad_(true),
            bias: Tensor::from_slice(
                crate::DType::F32,
                crate::Shape::new([out_features]),
                &bias_data,
            )
            .requires_grad_(true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Auto-device sync (Ensures x is on the same device as the weights)
        let x = x.to(self.weight.device.clone());

        if x.device.is_gpu() && self.weight.device.is_gpu() && self.bias.device.is_gpu() {
            let ctx = match &self.weight.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };

            // 1. Flatten 3D [B, S, H] to 2D [B*S, H] for the fused shader
            let original_shape = x.shape.clone();
            let x_2d = if x.rank() == 3 {
                let b = x.shape.dims()[0];
                let s = x.shape.dims()[1];
                let h = x.shape.dims()[2];
                x.view(Shape::new([b * s, h]))
            } else {
                x.clone()
            };

            // 2. Dispatch the Fused Linear Shader (x @ w + b)
            let out_2d = GpuContext::fused_linear(&ctx, &x_2d, &self.weight, &self.bias);

            // 3. Attach the FusedLinearOp to the 2D tensors for perfect Autograd tracking
            let out_2d_tracked =
                if x_2d.requires_grad || self.weight.requires_grad || self.bias.requires_grad {
                    let op = Arc::new(crate::ops::linear_fused::FusedLinearOp {
                        x: x_2d.clone(),
                        w: self.weight.clone(), // <--- FIXED: The struct expects 'w'
                        bias: self.bias.clone(), // (If compiler complains here, change 'bias' to 'b')
                    });
                    out_2d.with_node(
                        op,
                        vec![x_2d.clone(), self.weight.clone(), self.bias.clone()],
                    )
                } else {
                    out_2d
                };

            // 4. Reshape back to 3D [B, S, N] using view()
            if original_shape.rank() == 3 {
                let b = original_shape.dims()[0];
                let s = original_shape.dims()[1];
                let n = self.weight.shape.dims()[1];
                out_2d_tracked.view(Shape::new([b, s, n]))
            } else {
                out_2d_tracked
            }
        } else {
            // CPU Fallback
            let out = x.matmul(&self.weight);
            out.add(&self.bias)
        }
    }

    pub fn to(&mut self, device: crate::Device) {
        self.weight = self.weight.to(device.clone());
        self.bias = self.bias.to(device.clone());
    }
}
