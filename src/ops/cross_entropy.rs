use crate::autograd::Op;
use crate::{DType, Tensor};

#[derive(Debug)]
pub struct CrossEntropyLoss {
    pub logits: Tensor,
    pub targets: Vec<u32>,
    pub softmax_probs: Vec<f32>,
    pub batch_size: usize,
    pub num_classes: usize,
}

impl Op for CrossEntropyLoss {
    // 1. Added the required name method for the Autograd Graph
    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }

    fn backward(&self, _grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grads = crate::kernels::loss::cross_entropy_backward(
            &self.softmax_probs,
            &self.targets,
            self.batch_size,
            self.num_classes,
        );

        let grad_tensor = Tensor::from_slice(DType::F32, self.logits.shape.clone(), &grads);

        // Logits get the gradient, Targets do not (they are discrete integers)
        vec![Some(grad_tensor), None]
    }
}
