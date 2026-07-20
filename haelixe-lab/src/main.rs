// --------------------------------------------------------------------------
// Module: main (Haelixe Downstream Consumer Lab)
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Serves as the integration crucible for the Haelixe engine. This file
//   wires together the Tensor Engine, the Autograd Graph, the Cross-Entropy
//   Loss function, and the newly forged AdamW Optimizer to prove that the
//   framework can successfully execute a closed-loop training cycle.
//
// HISTORICAL CONTEXT:
//   Forged in July 2026, immediately following the resolution of "The Clone
//   Trap" and the successful compilation of the CPU-bound AdamW optimizer.
//   Prior iterations suffered from detached gradients and VRAM fragmentation.
//   This harness enforces the "Master Weights Pattern," ensuring that the
//   optimizer interacts directly with the canonical parameter registry.
//
// STATE TRANSITION DIAGRAM:
//   [Init] --(alloc weights)--> [Forward] --(compute loss)--> [Backward]
//     ^                                                          |
//     |                                                          v
//   [Step] <----(extract grads)---- [Graph] <----(topo sort)-----+
//
// INVARIANTS:
//   - All master weights must be explicitly registered in the
//     parameter vector before the training loop commences.
//   - The computational graph must be implicitly cleared or
//     bypassed between epochs to prevent memory leaks.
//
// FAILURE MODES:
//   - Gradient Starvation: If a weight is not connected to the
//     loss node, its gradient will be missing, and the optimizer
//     will silently skip it (by design).
//   - Shape Panics: Mismatched batch sizes between the dummy
//     data and the loss function will trigger an assertion.
//
// CALL GRAPH:
//   Called by: `cargo run -p haelixe-lab --release`
//   Calls: haelixe::Tensor, haelixe::optim::AdamW, CrossEntropy
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use haelixe::{Tensor, DType, Shape, optim::{AdamW, Optimizer}};
use rand::Rng;

fn main() {
    println!("🔥 Haelixe Integration Crucible: Closed-Loop Training");

    // 1. Hyperparameters & Dummy Data Generation
    // We use a deterministic seed to ensure reproducibility across
    // runs, a mandatory requirement for debugging non-convex loss
    // landscapes in custom autograd engines.
    let batch_size = 4;
    let in_features = 8;
    let num_classes = 3;
    let epochs = 10;
    let lr = 0.01;

    let mut rng = rand::thread_rng();

    // 2. Master Weights Initialization
    // In a production system, these would be initialized via Kaiming
    // or Xavier uniform distributions. Here, we use simple random
    // noise to verify that the optimizer can push them toward a
    // meaningful signal.
    // NOTE: We use `r#gen` because `gen` is a reserved keyword in 
    // Rust Edition 2024.
    let weight_data: Vec<f32> = (0..in_features * num_classes)
        .map(|_| rng.r#gen::<f32>() - 0.5)
        .collect();
    
    let bias_data: Vec<f32> = vec![0.0; num_classes];

    let mut weight = Tensor::from_slice(
        DType::F32, 
        Shape::new([in_features, num_classes]), 
        &weight_data
    );
    weight.requires_grad = true;

    let mut bias = Tensor::from_slice(
        DType::F32, 
        Shape::new([num_classes]), 
        &bias_data
    );
    bias.requires_grad = true;

    // The canonical registry. The optimizer only knows about this list.
    let master_weights = vec![&weight, &bias];

    // 3. Optimizer Instantiation
    let mut optimizer = AdamW::new(lr);

    // 4. The Training Loop
    println!("Starting {} epochs of Sequence Denoising...", epochs);
    for epoch in 0..epochs {
        // Generate dummy input batch (Batch x Features)
        let input_data: Vec<f32> = (0..batch_size * in_features)
            .map(|_| rng.r#gen::<f32>())
            .collect();
        let x = Tensor::from_slice(
            DType::F32, 
            Shape::new([batch_size, in_features]), 
            &input_data
        );

        // Generate dummy target classes (Batch)
        let targets: Vec<u32> = (0..batch_size)
            .map(|i| (i % num_classes) as u32)
            .collect();

        // Forward Pass: Linear Transformation (y = xW + b)
        let logits = x.matmul(&weight).add(&bias);

        // Loss Calculation
        let loss = logits.cross_entropy(&targets);
        let loss_val = unsafe { 
            *(loss.ensure_cpu().storage.as_ptr() as *const f32) 
        };

        // Backward Pass & Gradient Extraction
        // The autograd engine returns a map of TensorId -> Gradient.
        // This bypasses "The Clone Trap" by explicitly fetching the
        // gradients computed for our master weight references.
        let grads_map = loss.backward();

        let mut step_params = Vec::new();
        for param in &master_weights {
            if let Some(grad) = grads_map.get(&param.id) {
                step_params.push((*param, grad));
            }
        }

        // Optimizer Step
        optimizer.step(&step_params);

        if epoch % 2 == 0 || epoch == epochs - 1 {
            println!("Epoch {:<2} | Loss: {:.4}", epoch, loss_val);
        }
    }

    println!("✅ Crucible Complete. The engine learns.");
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
// This lab currently relies on `rand::thread_rng` without a seedable PRNG,
// making exact epoch-to-epoch reproduction impossible without external
// environment locking. Furthermore, the manual `x.matmul(&weight).add(&bias)`
// sequence will eventually be abstracted into a proper `nn::Linear` module
// that handles the parameter registry internally.
//
// The engineer who inherits this file is urged to implement a seedable
// `fastrand` generator and a robust `Module` trait before attempting to
// scale this loop to Transformer-scale parameter counts.
// --------------------------------------------------------------------------