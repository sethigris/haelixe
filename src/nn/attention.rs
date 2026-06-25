use crate::{LayerNorm, Linear, Shape, Tensor};

pub struct MultiHeadAttention {
    pub norm: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        assert_eq!(
            hidden_dim % num_heads,
            0,
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;

        Self {
            norm: LayerNorm::new(hidden_dim, 1e-5),
            q_proj: Linear::new(hidden_dim, hidden_dim),
            k_proj: Linear::new(hidden_dim, hidden_dim),
            v_proj: Linear::new(hidden_dim, hidden_dim),
            out_proj: Linear::new(hidden_dim, hidden_dim),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let b = x.shape.dims()[0];
        let s = x.shape.dims()[1];
        let h = self.num_heads;
        let d = self.head_dim;
        let scale = 1.0 / (d as f32).sqrt();

        // Pre-LN: Normalize before projecting!
        let x_norm = self.norm.forward(x);

        let q = self.q_proj.forward(&x_norm);
        let k = self.k_proj.forward(&x_norm);
        let v = self.v_proj.forward(&x_norm);

        // 2. Split heads -> [B, S, H, D] -> Transpose to [B, H, S, D]
        let q = q.view(Shape::new([b, s, h, d])).transpose(1, 2);
        let k = k.view(Shape::new([b, s, h, d])).transpose(1, 2);
        let v = v.view(Shape::new([b, s, h, d])).transpose(1, 2);

        // 3. Flatten B and H to reuse 3D Batched MatMul -> [B*H, S, D]
        let q_flat = q.view(Shape::new([b * h, s, d]));
        let k_flat = k.view(Shape::new([b * h, s, d]));
        let v_flat = v.view(Shape::new([b * h, s, d]));

        // 4. Batched Attention Math (Zero CPU Loops!)
        // scores = (Q @ K^T) * scale
        let scores = q_flat.batched_matmul(&k_flat, true, scale);

        // Attention Weights = Softmax(Scores)
        let attn = scores.softmax();

        // Context = Attn @ V
        let ctx = attn.batched_matmul(&v_flat, false, 1.0);

        // 5. Reassemble heads -> [B*H, S, D] -> [B, H, S, D] -> [B, S, H, D] -> [B, S, Hidden]
        let out = ctx
            .view(Shape::new([b, h, s, d]))
            .transpose(1, 2)
            .view(Shape::new([b, s, h * d]));

        // 6. Final output projection
        self.out_proj.forward(&out)
    }
}
