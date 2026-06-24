use crate::{Op, Shape, Tensor};
#[derive(Debug)]
pub struct ViewOp {
    pub original_shape: Shape,
}
impl Op for ViewOp {
    fn name(&self) -> &'static str {
        "View"
    }
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // The gradient just gets reshaped back to the original input shape!
        vec![Some(grad_output.view(self.original_shape.clone()))]
    }
}
