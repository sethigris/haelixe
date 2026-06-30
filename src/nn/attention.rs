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

        // 1. Pre-LN
        let x_norm = self.norm.forward(x);

        // 2. Project Q, K, V -> [B, S, Hidden]
        let q = self.q_proj.forward(&x_norm);
        let k = self.k_proj.forward(&x_norm);
        let v = self.v_proj.forward(&x_norm);

        // 3. Split heads -> [B, H, S, D] -> Flatten to [B*H, S, D]
        let mut q = q
            .view(Shape::new([b, s, h, d]))
            .transpose(1, 2)
            .view(Shape::new([b * h, s, d]));
        let mut k = k
            .view(Shape::new([b, s, h, d]))
            .transpose(1, 2)
            .view(Shape::new([b * h, s, d]));
        let v = v
            .view(Shape::new([b, s, h, d]))
            .transpose(1, 2)
            .view(Shape::new([b * h, s, d]));

        // 4. Apply RoPE strictly to Q and K
        let rope = crate::nn::rope::RoPE::new(d, 10000.0);
        q = rope.forward(&q);
        k = rope.forward(&k);

        // 5. Flash-Attention
        // Extract the exact device from the model weights.
        // Do not call Device::gpu() here, as it may spawn a conflicting logical device.
        let gpu_device = self.q_proj.weight.device.clone();
        let q_gpu = q.to(gpu_device.clone());
        let k_gpu = k.to(gpu_device.clone());
        let v_gpu = v.to(gpu_device.clone());

        let ctx = q_gpu.flash_attention(&k_gpu, &v_gpu, scale);

        // 6. Reassemble heads -> [B*H, S, D] -> [B, H, S, D] -> [B, S, H, D] -> [B, S, Hidden]
        let out = ctx
            .view(Shape::new([b, h, s, d]))
            .transpose(1, 2)
            .view(Shape::new([b, s, h * d]));

        // 7. Final output projection
        self.out_proj.forward(&out)
    }

    pub fn to(&mut self, device: crate::Device) {
        self.norm.to(device.clone());
        self.q_proj.to(device.clone());
        self.k_proj.to(device.clone());
        self.v_proj.to(device.clone());
        self.out_proj.to(device.clone());
    }
}
