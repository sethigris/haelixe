use haelixe::{DType, Device, RMSNorm, Shape, Tensor, TransformerBlock, optim::AdamW};
use std::f32::consts::PI;

fn main() {
    println!("Haelixe Downstream Consumer Test: Sequence Denoising");
    println!("--------------------------------------------------");

    let gpu = Device::gpu();
    let batch_size = 4;
    let seq_len = 32;
    let hidden_dim = 64;
    let num_heads = 4;

    // 1. Initialize the Architecture
    let mut embed = haelixe::Linear::new(1, hidden_dim);
    let mut block = TransformerBlock::new(hidden_dim, num_heads);
    let mut final_norm = RMSNorm::new(hidden_dim);
    let mut head = haelixe::Linear::new(hidden_dim, 1);

    // Migrate all internal parameters to the GPU in-place
    embed.to(gpu.clone());
    block.to(gpu.clone());
    final_norm.to(gpu.clone());
    head.to(gpu.clone());

    // Collect parameter references for the optimizer
    let params = vec![
        embed.weight.clone(),
        embed.bias.clone(),
        head.weight.clone(),
        head.bias.clone(),
        final_norm.weight.clone(),
        // Unlike standard LayerNorm which utilizes both a scaling factor (gamma/weight)
        // and a translation factor (beta/bias), RMSNorm mathematically eliminates
        // the translation step to prevent variance collapse. We only collect the
        // scaling weights here, as the bias field intentionally does not exist.
        block.norm1.weight.clone(),
        block.norm2.weight.clone(),
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

    let mut optimizer = AdamW::new(0.001); // Base LR, will be overridden by schedule

    let max_lr = 0.001; // Lowered from 0.005 to prevent attention entropy collapse
    let min_lr = 0.00005; // Tighter floor for precise convergence
    let total_epochs = 100; // Extended for smoother cosine decay convergence

    println!("Starting Training Loop with Cosine Annealing...\n");

    for epoch in 0..total_epochs {
        // MATHEMATICAL NOTE:
        // The Cosine Annealing formula smoothly interpolates the learning rate
        // between max_lr and min_lr. At epoch 0, the cosine is 1.0 (yielding max_lr).
        // At the final epoch, the cosine is -1.0 (yielding min_lr). This prevents
        // the destructive oscillation that causes networks to plateau prematurely.
        let cosine_decay = 0.5 * (1.0 + (PI * epoch as f32 / total_epochs as f32).cos());
        let current_lr = min_lr + 0.5 * (max_lr - min_lr) * cosine_decay;
        optimizer.set_lr(current_lr);

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
        let h = final_norm.forward(&h);
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
            println!(
                "Epoch {:<3} | LR: {:.6} | MSE Loss: {:.6}",
                epoch, current_lr, loss_val
            );
        }
    }

    println!("\nExperiment Complete.");
}
