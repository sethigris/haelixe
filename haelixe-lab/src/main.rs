// --------------------------------------------------------------------------
// Module: main (Softmax Battle-Test Crucible)
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Mathematically verifies the Softmax engine across 5 stages:
//   1. Forward pass correctness (known values)
//   2. Numerical stability (LogSumExp with extreme logits)
//   3. Probability simplex invariant (rows sum to 1.0)
//   4. Backward pass Jacobian correctness
//   5. Autograd graph integration
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-22
// --------------------------------------------------------------------------
use haelixe::{DType, Shape, Tensor};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn main() {
    println!("========================================");
    println!("  HAELIXE SOFTMAX BATTLE-TEST CRUCIBLE");
    println!("========================================\n");

    let mut all_passed = true;

    // ========================================
    // STAGE 1: Forward Pass Correctness
    // ========================================
    println!("--- STAGE 1: Forward Pass (Known Values) ---");
    {
        // Input: [1.0, 2.0, 3.0]
        // Expected: exp([1,2,3] - 3) / sum(exp([1,2,3] - 3))
        //         = exp([-2,-1,0]) / (exp(-2)+exp(-1)+exp(0))
        //         = [0.1353, 0.3679, 1.0] / 1.5032
        //         = [0.0900, 0.2447, 0.6652]

        // FIX: Added f32 suffix to force 32-bit memory allocation
        let x = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[1.0f32, 2.0, 3.0]);

        let y = x.softmax();
        let y_cpu = y.ensure_cpu();
        let y_data = unsafe { std::slice::from_raw_parts(y_cpu.storage.as_ptr() as *const f32, 3) };
        let expected = [0.0900, 0.2447, 0.6652];
        let pass = approx_eq(y_data[0], expected[0], 1e-3)
            && approx_eq(y_data[1], expected[1], 1e-3)
            && approx_eq(y_data[2], expected[2], 1e-3);
        println!("  Input:    [1.0, 2.0, 3.0]");
        println!(
            "  Output:   [{:.4}, {:.4}, {:.4}]",
            y_data[0], y_data[1], y_data[2]
        );
        println!(
            "  Expected: [{:.4}, {:.4}, {:.4}]",
            expected[0], expected[1], expected[2]
        );
        println!("  Result:   {}\n", if pass { "PASS" } else { "FAIL" });
        if !pass {
            all_passed = false;
        }
    }

    // ========================================
    // STAGE 2: Numerical Stability (LogSumExp)
    // ========================================
    println!("--- STAGE 2: Numerical Stability (Extreme Logits) ---");
    {
        // Naive softmax would compute exp(1000) = Infinity -> NaN
        // LogSumExp trick subtracts max first, keeping values stable

        // FIX: Added f32 suffix
        let x = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[1000.0f32, 1001.0, 1002.0]);

        let y = x.softmax();
        let y_cpu = y.ensure_cpu();
        let y_data = unsafe { std::slice::from_raw_parts(y_cpu.storage.as_ptr() as *const f32, 3) };
        let has_nan = y_data.iter().any(|v| v.is_nan());
        let has_inf = y_data.iter().any(|v| v.is_infinite());
        let pass = !has_nan && !has_inf;
        println!("  Input:    [1000.0, 1001.0, 1002.0]");
        println!(
            "  Output:   [{:.4}, {:.4}, {:.4}]",
            y_data[0], y_data[1], y_data[2]
        );
        println!("  NaN: {} | Inf: {}", has_nan, has_inf);
        println!("  Result:   {}\n", if pass { "PASS" } else { "FAIL" });
        if !pass {
            all_passed = false;
        }
    }

    // ========================================
    // STAGE 3: Probability Simplex Invariant
    // ========================================
    println!("--- STAGE 3: Probability Simplex (Rows Sum to 1.0) ---");
    {
        // FIX: Added f32 suffix
        let x = Tensor::from_slice(
            DType::F32,
            Shape::new([3, 4]),
            &[
                -1.0f32, 0.0, 1.0, 2.0, 3.0, 1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let y = x.softmax();
        let y_cpu = y.ensure_cpu();
        let y_data =
            unsafe { std::slice::from_raw_parts(y_cpu.storage.as_ptr() as *const f32, 12) };
        let mut pass = true;
        for row in 0..3 {
            let sum: f32 = (0..4).map(|c| y_data[row * 4 + c]).sum();
            let row_pass = approx_eq(sum, 1.0, 1e-5);
            println!(
                "  Row {} sum: {:.6} {}",
                row,
                sum,
                if row_pass { "OK" } else { "FAIL" }
            );
            if !row_pass {
                pass = false;
            }
        }
        // Also check all values are in [0, 1]
        let all_valid = y_data.iter().all(|v| *v >= 0.0 && *v <= 1.0);
        println!("  All values in [0,1]: {}", all_valid);
        if !all_valid {
            pass = false;
        }
        println!("  Result:   {}\n", if pass { "PASS" } else { "FAIL" });
        if !pass {
            all_passed = false;
        }
    }

    // ========================================
    // STAGE 4: Backward Pass Jacobian
    // ========================================
    println!("--- STAGE 4: Backward Pass (Jacobian Correctness) ---");
    {
        // For softmax, dx_i = y_i * (grad_i - sum(grad_j * y_j))
        // Input: [1.0, 2.0, 3.0] -> y = [0.0900, 0.2447, 0.6652]
        // Grad:  [1.0, 0.0, 0.0]
        // dot = 1.0*0.0900 + 0.0*0.2447 + 0.0*0.6652 = 0.0900
        // dx_0 = 0.0900 * (1.0 - 0.0900) = 0.0819
        // dx_1 = 0.2447 * (0.0 - 0.0900) = -0.0220
        // dx_2 = 0.6652 * (0.0 - 0.0900) = -0.0599

        // FIX: Added f32 suffixes
        let mut x = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[1.0f32, 2.0, 3.0]);
        x.requires_grad = true;
        let y = x.softmax();

        // Manually trigger backward with a seed gradient
        let grad_seed = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[1.0f32, 0.0, 0.0]);
        let grads_map = y.backward_with_seed(&grad_seed);

        let pass = if let Some(dx) = grads_map.get(&x.id) {
            let dx_cpu = dx.ensure_cpu();
            let dx_data =
                unsafe { std::slice::from_raw_parts(dx_cpu.storage.as_ptr() as *const f32, 3) };
            let expected = [0.0819, -0.0220, -0.0599];
            let ok = approx_eq(dx_data[0], expected[0], 1e-3)
                && approx_eq(dx_data[1], expected[1], 1e-3)
                && approx_eq(dx_data[2], expected[2], 1e-3);
            println!("  Grad:     [1.0, 0.0, 0.0]");
            println!(
                "  dx:       [{:.4}, {:.4}, {:.4}]",
                dx_data[0], dx_data[1], dx_data[2]
            );
            println!(
                "  Expected: [{:.4}, {:.4}, {:.4}]",
                expected[0], expected[1], expected[2]
            );
            ok
        } else {
            println!("  ERROR: No gradient found for input tensor!");
            false
        };
        println!("  Result:   {}\n", if pass { "PASS" } else { "FAIL" });
        if !pass {
            all_passed = false;
        }
    }

    // ========================================
    // STAGE 5: Autograd Graph Integration
    // ========================================
    println!("--- STAGE 5: Autograd Graph Integration ---");
    {
        // Verify softmax works inside a larger computation graph
        // loss = sum(softmax(x)) should have gradient = 0 for all inputs
        // because sum(softmax(x)) = 1.0 (constant), so d/dx = 0

        // FIX: Added f32 suffix
        let mut x = Tensor::from_slice(DType::F32, Shape::new([1, 4]), &[1.0f32, 2.0, 3.0, 4.0]);
        x.requires_grad = true;
        let y = x.softmax();
        let loss = y.sum();
        let grads_map = loss.backward();

        let pass = if let Some(dx) = grads_map.get(&x.id) {
            let dx_cpu = dx.ensure_cpu();
            let dx_data =
                unsafe { std::slice::from_raw_parts(dx_cpu.storage.as_ptr() as *const f32, 4) };
            let all_near_zero = dx_data.iter().all(|v| v.abs() < 1e-4);
            println!(
                "  d/dx sum(softmax(x)): [{:.6}, {:.6}, {:.6}, {:.6}]",
                dx_data[0], dx_data[1], dx_data[2], dx_data[3]
            );
            println!("  All near zero (sum=const): {}", all_near_zero);
            all_near_zero
        } else {
            println!("  ERROR: No gradient found!");
            false
        };
        println!("  Result:   {}\n", if pass { "PASS" } else { "FAIL" });
        if !pass {
            all_passed = false;
        }
    }

    // ========================================
    // FINAL VERDICT
    // ========================================
    println!("========================================");
    if all_passed {
        println!("  ALL 5 STAGES PASSED");
        println!("  Softmax Engine is MATHEMATICALLY PURE");
    } else {
        println!("  SOME STAGES FAILED");
        println!("  Review output above for details");
    }
    println!("========================================");
}
