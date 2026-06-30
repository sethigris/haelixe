use crate::{Linear, Tensor};

pub struct FeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl FeedForward {
    pub fn new(hidden_dim: usize) -> Self {
        let ff_dim = hidden_dim * 4; // Standard 4x expansion
        Self {
            linear1: Linear::new(hidden_dim, ff_dim),
            linear2: Linear::new(ff_dim, hidden_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to(self.linear1.weight.device.clone());
        let h = self.linear1.forward(&x);
        let h = h.gelu();
        self.linear2.forward(&h)
    }

    pub fn to(&mut self, device: crate::Device) {
        self.linear1.to(device.clone());
        self.linear2.to(device.clone());
    }
}
