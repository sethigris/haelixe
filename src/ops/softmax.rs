use crate::{Tensor, autograd::Op};

#[derive(Debug)]
pub struct SoftmaxOp {
    // CRITICAL: We store the OUTPUT (y), not the input (x).
    // The Jacobian of Softmax requires the probabilities, not the logits.
    pub output: Tensor, 
}

impl Op for SoftmaxOp {
    fn name(&self) -> &'static str { "Softmax" }
    
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let dx = crate::kernels::softmax::softmax_backward(&self.output, grad);
        vec![Some(dx)]
    }
}
