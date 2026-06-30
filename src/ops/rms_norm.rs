use crate::{Op, Tensor};

#[derive(Debug)]
pub struct RMSNormOp {
    pub x: Tensor,
    pub weight: Tensor,
    pub eps: f32,
}

impl Op for RMSNormOp {
    fn name(&self) -> &'static str {
        "RMSNorm"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let (dx, dw) =
            crate::kernels::rms_norm_backward(grad_output, &self.x, &self.weight, self.eps);
        vec![Some(dx), Some(dw)]
    }
}
