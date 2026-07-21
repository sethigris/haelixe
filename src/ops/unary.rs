use crate::{Tensor, autograd::Op};

#[derive(Debug)]
pub struct UnaryOp {
    pub x: Tensor,
    pub op_code: u32,
}

impl Op for UnaryOp {
    fn name(&self) -> &'static str {
        "Unary"
    }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let dx = crate::kernels::unary::backward_cpu(&self.x, grad, self.op_code);
        vec![Some(dx)]
    }
}
