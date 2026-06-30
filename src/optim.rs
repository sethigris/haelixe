use crate::{DType, Device, GpuContext, Tensor, TensorId};
use std::collections::HashMap;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: u32,
    // Maps TensorId to (Momentum Tensor, Variance Tensor)
    pub state: HashMap<TensorId, (Tensor, Tensor)>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self, weights_and_grads: &[(&Tensor, &Tensor)]) {
        self.step_count += 1;

        // Bias correction math
        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);
        let params: [f32; 8] = [
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            bc1,
            bc2,
            0.0,
        ];

        for (weight, grad) in weights_and_grads {
            // 1. Lazily allocate m and v on the FIRST step
            if !self.state.contains_key(&weight.id) {
                let m = Tensor::zeros(DType::F32, weight.shape.clone()).to(weight.device.clone());
                let v = Tensor::zeros(DType::F32, weight.shape.clone()).to(weight.device.clone());
                self.state.insert(weight.id, (m, v));
            }

            let (m, v) = self.state.get(&weight.id).unwrap();

            // 2. Dispatch the Fused WGSL Shader
            if weight.device.is_gpu() {
                let ctx = match &weight.device {
                    Device::Gpu(c) => c.clone(),
                    _ => unreachable!(),
                };

                // Ensure grad is on GPU
                // `.to()` automatically clones if already on GPU, or transfers if on CPU!
                let grad_gpu = grad.to(weight.device.clone());

                GpuContext::adamw_step_gpu(&ctx, weight, &grad_gpu, m, v, params);
            } else {
                panic!("CPU AdamW not implemented for this engineering demo!");
            }
        }
    }

    // In high-dimensional, non-convex loss landscapes, a static learning rate
    // prevents the optimizer from settling into sharp global minima. This method
    // allows the external training loop to inject dynamic learning rate schedules
    // (like Cosine Annealing or Linear Warmup) without reconstructing the optimizer
    // state or destroying the accumulated momentum (m) and variance (v) buffers.
    pub fn set_lr(&mut self, new_lr: f32) {
        self.lr = new_lr;
    }
}
