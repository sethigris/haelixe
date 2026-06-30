use crate::{Op, Tensor};

#[derive(Debug)]
pub struct GELUOp {
    pub input: Tensor,
}

impl Op for GELUOp {
    fn name(&self) -> &'static str {
        "GELU"
    }
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(crate::kernels::gelu_backward(
            grad_output,
            &self.input,
        ))]
    }
}
