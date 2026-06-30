use crate::{Op, Tensor};

#[derive(Debug)]
pub struct MSELossOp {
    pub pred: Tensor,
    pub target: Tensor,
}

impl Op for MSELossOp {
    fn name(&self) -> &'static str {
        "MSELoss"
    }

    fn backward(&self, _grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let dx = crate::kernels::mse_loss_backward(&self.pred, &self.target);
        // Gradient flows to pred; target is treated as a constant (None)
        vec![Some(dx), None]
    }
}
