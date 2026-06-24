use axiom::{DType, MultiHeadAttention, Shape, Tensor};
use std::time::Instant;

fn main() {
    println!("Testing Multi-Head Attention (Transformer)\n");

    let batch_size = 2;
    let seq_len = 16;
    let hidden_dim = 64;
    let num_heads = 4;

    // Dummy input: [Batch, Sequence, Hidden]
    let x_data: Vec<f32> = (0..batch_size * seq_len * hidden_dim)
        .map(|i| (i % 10) as f32 * 0.01)
        .collect();
    let x = Tensor::from_slice(
        DType::F32,
        Shape::new([batch_size, seq_len, hidden_dim]),
        &x_data,
    )
    .requires_grad_(true);

    let mha = MultiHeadAttention::new(hidden_dim, num_heads);

    println!("Running MHA Forward Pass...");
    let start = Instant::now();
    let out = mha.forward(&x);
    println!("Forward pass took: {:?}", start.elapsed());
    println!("Output shape: {:?} (Should be [2, 16, 64])", out.shape);

    println!("\nRunning MHA Backward Pass...");
    let loss = out.sum();
    let start = Instant::now();
    let grads = loss.backward();
    println!("Backward pass took: {:?}", start.elapsed());

    let x_grad = grads.get(&x.id).unwrap();
    println!("Input Gradient shape: {:?}", x_grad.shape);
}
