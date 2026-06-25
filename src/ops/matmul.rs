use crate::{Op, Tensor};

#[derive(Debug)]
pub struct MatMulOp {
    pub a: Tensor,
    pub b: Tensor,
}

impl Op for MatMulOp {
    fn name(&self) -> &'static str {
        "MatMul"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let a = &self.a;
        let b = &self.b;

        if a.rank() == 3 && b.rank() == 3 {
            // 3D Batched MatMul backward pass (Routes to the GPU Batched Shader!)
            // grad_a = grad_out @ B^T
            let grad_a = grad_output.batched_matmul(b, true, 1.0);
            // grad_b = A^T @ grad_out
            let grad_b = a.transpose(1, 2).batched_matmul(grad_output, false, 1.0);

            vec![Some(grad_a), Some(grad_b)]
        } else {
            // 2D Fallback
            let a_cpu = a.ensure_cpu();
            let b_cpu = b.ensure_cpu();
            let grad_out_cpu = grad_output.ensure_cpu();

            let grad_a = grad_out_cpu.matmul(&b_cpu.t());
            let grad_b = a_cpu.t().matmul(&grad_out_cpu);

            vec![Some(grad_a), Some(grad_b)]
        }
    }
}
