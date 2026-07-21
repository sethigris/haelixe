use crate::{Shape, Tensor, autograd::Op};

#[derive(Debug)]
pub struct ReduceOp {
    pub orig_shape: Shape,
    pub op_code: u32, // 0 = Sum, 1 = Mean
}

impl Op for ReduceOp {
    fn name(&self) -> &'static str {
        "Reduce"
    }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let dx = crate::kernels::reduce::reduce_backward(grad, &self.orig_shape, self.op_code);
        vec![Some(dx)]
    }
}
