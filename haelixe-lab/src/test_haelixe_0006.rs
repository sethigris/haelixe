// --------------------------------------------------------------------------
// Module: main (Haelixe Downstream Consumer Lab)
// --------------------------------------------------------------------------
// PURPOSE:
//   Validates the `nn::Module` abstraction by training a Linear layer
//   using the composable parameter extraction API.
// --------------------------------------------------------------------------

use haelixe::{
    DType, Shape, Tensor,
    nn::{Linear, Module},
    optim::{AdamW, Optimizer},
};
use rand::Rng;

fn main() {
    println!(" Haelixe Crucible: Composable Module Training");

    let batch_size = 4;
    let in_features = 8;
    let num_classes = 3;
    let epochs = 20;
    let lr = 0.05; // Increased LR for faster convergence on this toy problem

    let mut rng = rand::thread_rng();

    // 1. Instantiate the Model
    let model = Linear::new(in_features, num_classes);

    // 2. Optimizer Instantiation
    let mut optimizer = AdamW::new(lr);

    println!("Starting {} epochs...", epochs);
    for epoch in 0..epochs {
        let input_data: Vec<f32> = (0..batch_size * in_features)
            .map(|_| rng.r#gen::<f32>())
            .collect();
        let x = Tensor::from_slice(
            DType::F32,
            Shape::new([batch_size, in_features]),
            &input_data,
        );

        let targets: Vec<u32> = (0..batch_size).map(|i| (i % num_classes) as u32).collect();

        // Forward Pass via Module API
        let logits = model.forward(&x);
        let loss = logits.cross_entropy(&targets);
        let loss_val = unsafe { *(loss.ensure_cpu().storage.as_ptr() as *const f32) };

        // Backward Pass
        let grads_map = loss.backward();

        // Extract parameters dynamically from the Module trait!
        let step_params: Vec<(&Tensor, &Tensor)> = model
            .parameters()
            .iter()
            .filter_map(|&p| grads_map.get(&p.id).map(|g| (p, g)))
            .collect();

        optimizer.step(&step_params);

        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("Epoch {:<2} | Loss: {:.4}", epoch, loss_val);
        }
    }

    println!("Module API Validated.");
}
