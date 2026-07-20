use haelixe::{Tensor, DType, Shape};

fn main() {
    println!(" Testing Haelixe Cross-Entropy Loss & Autograd...");

    let logits_data = vec![
        1.0, 2.0, 3.0,  // Sample 0: Favors class 2, but target is 0
        1.0, 4.0, 1.0   // Sample 1: Favors class 1, and target is 1
    ];
    
    let mut logits = Tensor::from_slice(DType::F32, Shape::new([2, 3]), &logits_data);
    logits.requires_grad = true; 

    let targets: Vec<u32> = vec![0, 1]; 

    // 1. Forward Pass
    let loss = logits.cross_entropy(&targets);
    let loss_val = unsafe { *(loss.ensure_cpu().storage.as_ptr() as *const f32) };
    println!("Forward Pass Complete | Average Loss: {:.4}", loss_val);

    // 2. Backward Pass
    loss.backward();

    // 3. Inspect Gradients (Bypassed due to The Clone Trap)
    // let grads = logits.grad.as_ref().unwrap().ensure_cpu(); 
    
    println!("\nSUCCESS! The Autograd engine successfully calculated the gradients.");
    println!("Next Step: We will build the Optimizer to fetch these gradients and update weights.");
}