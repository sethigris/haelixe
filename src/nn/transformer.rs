use crate::{FeedForward, LayerNorm, MultiHeadAttention, Tensor};

pub struct TransformerBlock {
    pub norm1: LayerNorm,
    pub mha: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub mlp: FeedForward,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            norm1: LayerNorm::new(hidden_dim, 1e-5),
            mha: MultiHeadAttention::new(hidden_dim, num_heads),
            norm2: LayerNorm::new(hidden_dim, 1e-5),
            mlp: FeedForward::new(hidden_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Pre-LN Attention with Residual Connection
        // x = x + MHA(LayerNorm(x))
        let normed_x = self.norm1.forward(x);
        let attn_out = self.mha.forward(&normed_x);
        let x = x.add(&attn_out);

        // 2. Pre-LN MLP with Residual Connection
        // x = x + MLP(LayerNorm(x))
        let normed_x2 = self.norm2.forward(&x);
        let mlp_out = self.mlp.forward(&normed_x2);
        let x = x.add(&mlp_out);

        x
    }

    pub fn to(&mut self, device: crate::Device) {
        self.norm1.to(device.clone());
        self.mha.to(device.clone());
        self.norm2.to(device.clone());
        self.mlp.to(device.clone());
    }
}
