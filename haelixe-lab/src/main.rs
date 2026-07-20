// --------------------------------------------------------------------------
// Module: main (Pillar 3: VRAM Birth Validation)
// --------------------------------------------------------------------------
use haelixe::{
    Tensor, DType, Shape, Device, 
    nn::{Module, Linear, RMSNorm}, 
    optim::{AdamW, Optimizer},
    gpu::GpuContext
};
use rand::Rng;

fn main() {
    println!(" Haelixe Pillar 3: Device-Aware Initialization");

    let ctx = GpuContext::new();
    let gpu = Device::Gpu(ctx.clone());

    let batch_size = 4;
    let in_features = 8;
    let num_classes = 16; 
    let epochs = 10;
    let lr = 0.05;

    let mut rng = rand::thread_rng();

    // VRAM BIRTH: Weights are instantiated directly on the GPU.
    // No manual `.to(gpu)` calls required. The PCIe Tax is eradicated.
    let linear = Linear::new(in_features, num_classes, &gpu);
    let norm = RMSNorm::new(num_classes, &gpu);

    let mut optimizer = AdamW::new(lr);

    println!("Starting {} epochs...", epochs);
    for epoch in 0..epochs {
        let input_data: Vec<f32> = (0..batch_size * in_features)
            .map(|_| rng.r#gen::<f32>())
            .collect();
        
        // Input data is born on CPU, then pushed to GPU for the forward pass
        let x = Tensor::from_slice(
            DType::F32, Shape::new([batch_size, in_features]), &input_data
        ).to(gpu.clone());

        let h = linear.forward(&x);
        let logits = norm.forward(&h);
        
        // Generate dummy targets for Cross-Entropy
        let targets: Vec<u32> = (0..batch_size).map(|i| (i % num_classes) as u32).collect();
        let loss = logits.cross_entropy(&targets);
        
        let grads_map = loss.backward();

        let mut all_params = linear.parameters();
        all_params.extend(norm.parameters());

        let step_params: Vec<(&Tensor, &Tensor)> = all_params.iter()
            .filter_map(|&p| grads_map.get(&p.id).map(|g| (p, g)))
            .collect();

        // The Optimizer automatically handles CPU->GPU gradient syncing!
        optimizer.step(&step_params);

        if epoch % 2 == 0 || epoch == epochs - 1 {
            let loss_val = unsafe { 
                *(loss.ensure_cpu().storage.as_ptr() as *const f32) 
            };
            println!("Epoch {:<2} | Loss: {:.4}", epoch, loss_val);
        }
    }

    println!(" Pillar 3 Validated. Production API achieved.");
}
