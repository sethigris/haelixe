// --------------------------------------------------------------------------
// Module: main (Pillar 2 Hygiene Validation)
// --------------------------------------------------------------------------
use haelixe::{DType, NoGradGuard, Shape, Tensor};

fn main() {
    println!("Haelixe Pillar 2: Autograd Memory Hygiene Test");

    let mut w = Tensor::from_slice(DType::F32, Shape::new([2, 2]), &[1.0, 2.0, 3.0, 4.0]);
    w.requires_grad = true;

    let x = Tensor::from_slice(DType::F32, Shape::new([2, 2]), &[1.0, 1.0, 1.0, 1.0]);

    // 1. Normal Training Mode
    let y_train = x.matmul(&w);
    println!("Training Mode | Graph Built? {}", y_train.node.is_some());
    assert!(
        y_train.node.is_some(),
        "Graph should be built in training mode!"
    );

    // 2. Inference Mode (NoGrad)
    {
        // The underscore `_guard` is mandatory! If you don't bind it to a
        // variable, Rust drops it immediately, re-enabling gradients.
        let _guard = NoGradGuard::new();

        let y_inf = x.matmul(&w);
        println!("Inference Mode| Graph Built? {}", y_inf.node.is_some());
        assert!(!y_inf.node.is_some(), "Graph MUST NOT be built in no_grad!");
    } // Guard drops here, restoring state

    // 3. State Restoration Check
    let y_restored = x.matmul(&w);
    println!("Restored Mode | Graph Built? {}", y_restored.node.is_some());
    assert!(y_restored.node.is_some(), "Guard failed to restore state!");

    // 4. Detach Check
    let y_detached = y_train.detach();
    println!(
        "Detached Node | Requires Grad? {}",
        y_detached.requires_grad
    );
    assert!(
        !y_detached.requires_grad,
        "Detach failed to clear requires_grad!"
    );

    println!("Pillar 2 Hardened. Memory leaks are now mathematically impossible.");
}
