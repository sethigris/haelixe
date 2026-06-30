use crate::nn::rms_norm::RMSNorm;
use crate::{FeedForward, MultiHeadAttention, Tensor};

pub struct TransformerBlock {
    // SYSTEMS ENGINEERING NOTE:
    // A Transformer block is fundamentally a sequence of residual transformations.
    // By deploying RMSNorm instead of LayerNorm, we reduce the computational overhead
    // (eliminating one full reduction pass for the mean) while mathematically
    // guaranteeing that the gradient signal flowing backward through the residual
    // connections remains unimpeded by mean-subtraction artifacts.
    pub norm1: RMSNorm,
    pub mha: MultiHeadAttention,
    pub norm2: RMSNorm,
    pub mlp: FeedForward,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            norm1: RMSNorm::new(hidden_dim),
            mha: MultiHeadAttention::new(hidden_dim, num_heads),
            norm2: RMSNorm::new(hidden_dim),
            mlp: FeedForward::new(hidden_dim),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Pre-Norm Architecture with Residual Connections

        // 1. Attention Block
        let norm1_out = self.norm1.forward(x);
        let attn_out = self.mha.forward(&norm1_out);
        let x = x.add(&attn_out); // Residual Connection

        // 2. Feed-Forward Block
        let norm2_out = self.norm2.forward(&x);
        let mlp_out = self.mlp.forward(&norm2_out);
        let x = x.add(&mlp_out); // Residual Connection

        x
    }

    pub fn to(&mut self, device: crate::Device) {
        self.norm1.to(device.clone());
        self.mha.to(device.clone());
        self.norm2.to(device.clone());
        self.mlp.to(device.clone());
    }
}
