use crate::{Op, Tensor};

#[derive(Debug)]
pub struct LayerNormOp {
    pub x: Tensor,
    pub weight: Tensor,
    pub mean: Tensor,
    pub rstd: Tensor,
    pub normalized_shape: usize,
}

impl Op for LayerNormOp {
    fn name(&self) -> &'static str {
        "LayerNorm"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let (dx, dweight, dbias) = crate::kernels::layer_norm_backward(
            grad_output,
            &self.x,
            &self.weight,
            &self.mean,
            &self.rstd,
            self.normalized_shape,
        );
        vec![Some(dx), Some(dweight), Some(dbias)]
    }
}
