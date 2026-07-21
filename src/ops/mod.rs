pub mod add;
pub mod cat;
pub mod contiguous;
pub mod cross_entropy;
pub mod flash_attention;
pub mod gelu;
pub mod layer_norm;
pub mod linear_fused;
pub mod matmul;
pub mod mse_loss;
pub mod relu;
pub mod rms_norm;
pub mod rope;
pub mod scalar_mul;
pub mod slice;
pub mod softmax;
pub mod sum;
pub mod transpose;
pub mod view;

use crate::{Op, Shape, Tensor};

#[derive(Debug)]
pub struct AddOp {
    pub a_shape: Shape,
    pub b_shape: Shape,
}

impl Op for AddOp {
    fn name(&self) -> &'static str {
        "Add"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let mut grad_a = grad_output.clone();
        let mut grad_b = grad_output.clone();

        // Reduce broadcasted dimensions
        while grad_a.rank() > self.a_shape.rank() {
            grad_a = crate::kernels::sum_axis(&grad_a, 0);
        }
        while grad_b.rank() > self.b_shape.rank() {
            grad_b = crate::kernels::sum_axis(&grad_b, 0);
        }

        // Note: sum_axis is CPU-only for now. In a full implementation,
        // we'd have GPU sum_axis too. For now, gradients flow back to CPU
        // for reduction, which is acceptable since gradient tensors are
        // typically much smaller than activations.

        vec![Some(grad_a), Some(grad_b)]
    }
}
pub mod binary_broadcast;
pub mod reduce;
