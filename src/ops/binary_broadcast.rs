use crate::{Tensor, autograd::Op};

#[derive(Debug)]
pub struct BinaryBroadcastOp {
    pub a: Tensor,
    pub b: Tensor,
    pub op_code: u32,
    pub out_shape: Vec<usize>,
    pub strides_a: Vec<usize>,
    pub strides_b: Vec<usize>,
}

impl Op for BinaryBroadcastOp {
    fn name(&self) -> &'static str {
        "BinaryBroadcast"
    }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let (da, db) = crate::kernels::broadcast::backward_cpu(
            grad,
            &self.a,
            &self.b,
            self.op_code,
            &self.out_shape,
            &self.strides_a,
            &self.strides_b,
        );
        vec![Some(da), Some(db)]
    }
}
