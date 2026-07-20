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
    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }

    fn backward(&self, _grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // 1. Compute the raw gradients (Softmax - OneHot)
        let grads = crate::kernels::loss::cross_entropy_backward(
            &self.softmax_probs,
            &self.targets,
            self.batch_size,
            self.num_classes,
        );

        // INTERCEPT: Print the gradients directly from the engine core
        println!("\n CROSS-ENTROPY BACKWARD PASS EXECUTED!");
        println!("Raw Gradients w.r.t Logits:");
        for i in 0..self.batch_size {
            let start = i * self.num_classes;
            let _end = start + self.num_classes;
            println!(
                "Sample {} (Target={}): [{:.4}, {:.4}, {:.4}]",
                i,
                self.targets[i],
                grads[start],
                grads[start + 1],
                grads[start + 2]
            );
        }

        // 2. Wrap in a Tensor to pass back down the graph
        let grad_tensor = Tensor::from_slice(DType::F32, self.logits.shape.clone(), &grads);

        vec![Some(grad_tensor), None]
    }
}
