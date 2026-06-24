use crate::{Op, Tensor};

#[derive(Debug)]
pub struct ScalarMulOp {
    pub scalar: f32,
}

impl Op for ScalarMulOp {
    fn name(&self) -> &'static str {
        "ScalarMul"
    }
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Chain rule: just multiply the incoming gradient by the same scalar!
        vec![Some(crate::kernels::scalar_mul(grad_output, self.scalar))]
    }
}
