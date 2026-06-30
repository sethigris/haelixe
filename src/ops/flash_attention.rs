use crate::{Op, Tensor};

#[derive(Debug)]
pub struct FlashAttentionOp {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub scale: f32,
}

impl Op for FlashAttentionOp {
    fn name(&self) -> &'static str {
        "FlashAttention"
    }

    fn backward(&self, grad_out: &Tensor) -> Vec<Option<Tensor>> {
        // Gradient Checkpointing: Recompute P to save VRAM!
        let scores = self.q.batched_matmul(&self.k, true, self.scale);
        let p = scores.softmax();

        // dV = P^T @ dO
        let grad_v = p.transpose(1, 2).batched_matmul(grad_out, false, 1.0);

        // dP = dO @ V^T
        let grad_p = grad_out.batched_matmul(&self.v, true, 1.0);

        // Backprop through Softmax using our optimized parallel kernel!
        let grad_scores = crate::kernels::softmax_backward(&grad_p, &p);

        // dQ = grad_scores @ K
        let grad_q = grad_scores.batched_matmul(&self.k, false, 1.0);

        // dK = grad_scores^T @ Q
        let grad_k = grad_scores
            .transpose(1, 2)
            .batched_matmul(&self.q, false, 1.0);

        vec![Some(grad_q), Some(grad_k), Some(grad_v)]
    }
}
