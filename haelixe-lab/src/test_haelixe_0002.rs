use haelixe::{DType, Shape, Tensor};

fn main() {
    println!("Haelixe Systems Test: BF16 Roundtrip Integrity");
    println!("--------------------------------------------");

    // 1. Generate a high-variance F32 tensor
    let data: Vec<f32> = (0..1000)
        .map(|i| (i as f32 * 0.12345).sin() * 100.0)
        .collect();
    let original = Tensor::from_slice(DType::F32, Shape::new([1000]), &data);

    println!("Original F32 Sample: {}", unsafe {
        original.storage.as_f32_slice()[0]
    });

    // 2. Cast down to BF16 (Memory footprint halves from 4000 bytes to 2000 bytes)
    let bf16_tensor = original.to_dtype(DType::BF16);
    println!(
        "BF16 Tensor Created. Byte size: {} (Expected ~2000)",
        bf16_tensor.storage.len()
    );

    // 3. Cast back to F32 for compute
    let roundtrip = bf16_tensor.to_dtype(DType::F32);
    let roundtrip_slice = unsafe { roundtrip.storage.as_f32_slice() };

    println!("Roundtrip F32 Sample: {}", roundtrip_slice[0]);

    // 4. Verify Mathematical Integrity
    let mut max_error: f32 = 0.0;
    for i in 0..1000 {
        let orig = unsafe { original.storage.as_f32_slice()[i] };
        let rt = roundtrip_slice[i];
        let error = (orig - rt).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("\nMaximum Roundtrip Error: {}", max_error);

    // BF16 has ~3 decimal digits of precision. For values around 100.0,
    // an error of ~0.5 to 1.0 is mathematically expected and acceptable.
    if max_error < 2.0 {
        println!("SUCCESS: BF16 memory layout and casting logic is mathematically sound.");
    } else {
        println!("FAILURE: Byte-level memory corruption detected during BF16 casting.");
    }
}

