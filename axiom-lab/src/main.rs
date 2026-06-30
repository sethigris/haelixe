use axiom::{DType, Device, Shape, Tensor, TransformerBlock, optim::AdamW};
use std::f32::consts::PI;

fn main() {
    println!("Axiom Downstream Consumer Test: Sequence Denoising");
    println!("--------------------------------------------------");

    let gpu = Device::gpu();
    let batch_size = 4;
    let seq_len = 32;
    let hidden_dim = 64;
    let num_heads = 4;

    // 1. Initialize the Architecture
    let mut embed = axiom::Linear::new(1, hidden_dim);
    let mut block = TransformerBlock::new(hidden_dim, num_heads);
    let mut head = axiom::Linear::new(hidden_dim, 1);

    // Migrate all internal parameters to the GPU in-place
    embed.to(gpu.clone());
    block.to(gpu.clone());
    head.to(gpu.clone());

    // Collect parameter references for the optimizer
    let params = vec![
        embed.weight.clone(),
        embed.bias.clone(),
        head.weight.clone(),
        head.bias.clone(),
        block.norm1.weight.clone(),
        block.norm1.bias.clone(),
        block.norm2.weight.clone(),
        block.norm2.bias.clone(),
        block.mha.q_proj.weight.clone(),
        block.mha.q_proj.bias.clone(),
        block.mha.k_proj.weight.clone(),
        block.mha.k_proj.bias.clone(),
        block.mha.v_proj.weight.clone(),
        block.mha.v_proj.bias.clone(),
        block.mha.out_proj.weight.clone(),
        block.mha.out_proj.bias.clone(),
        block.mlp.linear1.weight.clone(),
        block.mlp.linear1.bias.clone(),
        block.mlp.linear2.weight.clone(),
        block.mlp.linear2.bias.clone(),
    ];

    let mut optimizer = AdamW::new(0.005);

    println!("Starting Training Loop...\n");

    for epoch in 0..50 {
        // 2. Generate Synthetic Data (Clean Signal + Noise)
        let mut clean_data = vec![0.0f32; batch_size * seq_len];
        let mut noisy_data = vec![0.0f32; batch_size * seq_len];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let t = s as f32 / seq_len as f32;
                // Multi-frequency sine wave
                let clean = (2.0 * PI * t * 3.0).sin() + 0.5 * (2.0 * PI * t * 7.0).sin();
                // Pseudo-random noise
                let noise = ((b * seq_len + s) as f32 * 12.9898).sin() * 43758.5453 % 1.0 * 0.5;

                clean_data[b * seq_len + s] = clean;
                noisy_data[b * seq_len + s] = clean + noise;
            }
        }

        let x_noisy = Tensor::from_slice(
            DType::F32,
            Shape::new([batch_size, seq_len, 1]),
            &noisy_data,
        )
        .to(gpu.clone())
        .requires_grad_(true);

        let y_clean = Tensor::from_slice(
            DType::F32,
            Shape::new([batch_size, seq_len, 1]),
            &clean_data,
        )
        .to(gpu.clone());

        // 3. Forward Pass
        let h = embed.forward(&x_noisy);
        let h = block.forward(&h);
        let y_pred = head.forward(&h);

        // 4. Compute Loss (MSE)
        let loss = y_pred.mse_loss(&y_clean);

        // 5. Backward Pass
        let grads = loss.backward();

        // 6. Optimizer Step
        let mut step_params = Vec::new();
        for p in &params {
            if let Some(g) = grads.get(&p.id) {
                step_params.push((p, g));
            }
        }
        optimizer.step(&step_params);

        if epoch % 5 == 0 {
            let loss_cpu = loss.to(Device::Cpu);
            let loss_val = unsafe { *(loss_cpu.storage.as_ptr() as *const f32) };
            println!("Epoch {:<3} | MSE Loss: {:.6}", epoch, loss_val);
        }
    }

    println!("\nExperiment Complete.");
}
