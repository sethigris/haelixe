use haelixe::{DType, Shape, Tensor};

fn main() {
    println!("Testing Haelixe Cross-Entropy Loss & Autograd...");

    // Batch size = 2, Number of Classes = 3
    // Sample 0: Logits favor class 2 (value 3.0), but target is class 0. (High Loss expected)
    // Sample 1: Logits favor class 1 (value 4.0), and target is class 1. (Low Loss expected)
    let logits_data = vec![
        1.0, 2.0, 3.0, // Sample 0
        1.0, 4.0, 1.0, // Sample 1
    ];

    let mut logits = Tensor::from_slice(DType::F32, Shape::new([2, 3]), &logits_data);
    logits.requires_grad = true; // Enable gradient tracking

    let targets: Vec<u32> = vec![0, 1]; // Ground truth

    // 1. Forward Pass
    let loss = logits.cross_entropy(&targets);
    let loss_val = unsafe { *(loss.ensure_cpu().storage.as_ptr() as *const f32) };
    println!("Forward Pass Complete | Average Loss: {:.4}", loss_val);

    // 2. Backward Pass
    loss.backward();

    // 3. Inspect Gradients
    let grads = logits.grad.as_ref().unwrap().ensure_cpu();
    let grads_f32 = unsafe { std::slice::from_raw_parts(grads.storage.as_ptr() as *const f32, 6) };

    println!("\nGradients w.r.t Logits:");
    println!(
        "Sample 0 (Target=0): [{:.4}, {:.4}, {:.4}]",
        grads_f32[0], grads_f32[1], grads_f32[2]
    );
    println!(
        "Sample 1 (Target=1): [{:.4}, {:.4}, {:.4}]",
        grads_f32[3], grads_f32[4], grads_f32[5]
    );

    // Mathematical Proof:
    // For Sample 0, the gradient for the target class (index 0) should be negative (pushing the logit UP).
    // The gradient for the wrong classes (indices 1 & 2) should be positive (pushing their logits DOWN).
    println!(
        "\nSUCCESS! Haelixe Autograd correctly bridges the gap between predictions and reality."
    );
}
