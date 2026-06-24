use crate::{Op, Tensor};

#[derive(Debug)]
pub struct CatOp {
    pub n: usize,
    pub s: usize,
    pub d: usize,
}

impl Op for CatOp {
    fn name(&self) -> &'static str {
        "Cat"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::with_capacity(self.n);
        // Slice the 3D gradient back into N 2D gradients
        for i in 0..self.n {
            grads.push(Some(grad_output.get_2d_slice(i)));
        }
        grads
    }
}
