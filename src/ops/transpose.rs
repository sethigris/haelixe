use crate::{Op, Tensor};
#[derive(Debug)]
pub struct TransposeOp {
    pub dim1: usize,
    pub dim2: usize,
}
impl Op for TransposeOp {
    fn name(&self) -> &'static str {
        "Transpose"
    }
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // The gradient just gets transposed back!
        vec![Some(grad_output.transpose(self.dim1, self.dim2))]
    }
}
