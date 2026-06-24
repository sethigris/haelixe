use crate::{DType, Linear, Shape, Tensor};

pub struct MultiHeadAttention {
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

        // 1. Project Q, K, V -> [B, S, Hidden]
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 2. Split heads -> [B, S, H, D] -> Transpose to [B, H, S, D]
        let q = q.view(Shape::new([b, s, h, d])).transpose(1, 2);
        let k = k.view(Shape::new([b, s, h, d])).transpose(1, 2);
        let v = v.view(Shape::new([b, s, h, d])).transpose(1, 2);

        // 3. Flatten B and H to reuse 2D MatMul -> [B*H, S, D]
        let q_flat = q.view(Shape::new([b * h, s, d]));
        let k_flat = k.view(Shape::new([b * h, s, d]));
        let v_flat = v.view(Shape::new([b * h, s, d]));

        // 4. Allocate output buffer for attention results
        let out_flat = Tensor::empty(DType::F32, Shape::new([b * h, s, d]));

        // 4. Collect attention results in a Vec to preserve the Autograd graph!
        let mut ctx_list = Vec::with_capacity(b * h);

        // 5. Batched Attention Loop (Reusing our fast 2D Tiled GEMM!)
        for _i in 0..(b * h) {
            let q_i = q_flat.get_2d_slice(_i); // [S, D]
            let k_i = k_flat.get_2d_slice(_i); // [S, D]
            let v_i = v_flat.get_2d_slice(_i); // [S, D]

            // Scores = (Q @ K^T) * scale
            let scores = q_i.matmul(&k_i.t()).mul_scalar(scale);

            // Attention Weights = Softmax(Scores)
            let attn = scores.softmax();

            // Context = Attn @ V
            let ctx = attn.matmul(&v_i);

            ctx_list.push(ctx);
        }

        // 6. Concatenate (Autograd-aware!) and Reassemble heads
        let out_flat = Tensor::cat(&ctx_list); // [B*H, S, D]

        let out = out_flat
            .view(Shape::new([b, h, s, d]))
            .transpose(1, 2)
            .view(Shape::new([b, s, h * d]));

        // 7. Final output projection
        self.out_proj.forward(&out)
    }
}
