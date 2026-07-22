// tests/tensor_tests.rs
// Integration tests for the Haelixe Tensor Engine (#4) and Autograd (#8, #10)

use haelixe::*;
use std::collections::HashMap;

// -----------------------------------------------------------------
// Helper: extract f32 vector from a CPU tensor (safe for contiguous tensors)
// -----------------------------------------------------------------
fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.ensure_cpu();
    // We need the tensor in contiguous memory to read linearly.
    let contiguous = if cpu.is_contiguous() {
        cpu
    } else {
        cpu.contiguous()
    };
    assert_eq!(contiguous.dtype, DType::F32);
    let num_elements = contiguous.shape.num_elements();
    // SAFETY: the storage holds f32 bytes and the tensor is contiguous and on CPU.
    let ptr = contiguous.storage.as_ptr() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, num_elements).to_vec() }
}

// Helper: approximate equality for f32 tensors
fn assert_tensor_eq(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(a.shape.dims(), b.shape.dims(), "shape mismatch");
    let a_data = tensor_to_vec_f32(a);
    let b_data = tensor_to_vec_f32(b);
    for (x, y) in a_data.iter().zip(b_data.iter()) {
        assert!((x - y).abs() < tol, "values differ: {} vs {}", x, y);
    }
}

// ──────────────────────────────────────────────
// Tensor Creation & Basic Properties (#4)
// ──────────────────────────────────────────────
#[test]
fn test_zeros_and_ones() {
    let shape = Shape::new([2, 3]);
    let z = Tensor::zeros(DType::F32, shape.clone());
    let o = Tensor::ones(DType::F32, shape.clone());
    assert_eq!(z.shape.dims(), [2, 3]);
    assert_eq!(tensor_to_vec_f32(&z), vec![0.0; 6]);
    assert_eq!(tensor_to_vec_f32(&o), vec![1.0; 6]);
}

#[test]
fn test_from_slice_and_narrow() {
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::from_slice(DType::F32, Shape::new([3, 4]), &data);
    let view = t.narrow(1, 1, 2); // rows 0-2, cols 1..3
    let expected = vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0];
    assert_eq!(view.shape.dims(), [3, 2]);
    let v_cont = view.contiguous();
    assert_eq!(tensor_to_vec_f32(&v_cont), expected);
}

// BUG TEST: t() does NOT attach autograd node (should fail)
#[test]
fn test_transpose_2d_autograd() {
    let a = Tensor::from_slice(
        DType::F32,
        Shape::new([2, 3]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .requires_grad_(true);
    let b = a.t(); // supposed to track gradients, but currently lacks node
    assert!(b.requires_grad, "transpose should retain requires_grad");
    assert!(b.node.is_some(), "transpose t() must attach autograd node");
}

// BUG TEST: get_2d_slice on non-contiguous batch (fails)
#[test]
fn test_batched_slice_noncontiguous() {
    // Build a 2x3x4 tensor
    let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::from_slice(DType::F32, Shape::new([2, 3, 4]), &data);
    // Transpose dimensions 0 and 1 -> shape [3,2,4]
    let t_transposed = t.transpose(0, 1); // now batch dim is first
    // Now try to get "batch" index 1 along the new first dim
    let slice = t_transposed.get_2d_slice(1);
    // Expected: the 2x4 matrix at index 1 of the original dim 1
    let full_cont = t_transposed.contiguous();
    let full_data = tensor_to_vec_f32(&full_cont);
    let expected_start = 1 * 2 * 4; // offset in contiguous layout
    let expected: Vec<f32> = full_data[expected_start..expected_start + 8].to_vec();
    let slice_data = tensor_to_vec_f32(&slice.contiguous());
    assert_eq!(
        slice_data, expected,
        "get_2d_slice on non-contiguous tensor gives wrong data"
    );
}

// BUG TEST: cross_entropy with offset view (fails because of storage.len() misuse)
#[test]
fn test_cross_entropy_view_offset() {
    // Create a 2x3 tensor
    let data = (0..6).map(|x| x as f32).collect::<Vec<_>>();
    let big = Tensor::from_slice(DType::F32, Shape::new([2, 3]), &data).requires_grad_(true);
    // Slice out the second row only (batch size 1)
    let row = big.narrow(0, 1, 1); // shape [1,3], byte_offset shifted
    let targets = vec![1u32]; // class 1
    let loss = row.cross_entropy(&targets);
    // Build a clean tensor of the same row data to compare
    let row_data = vec![3.0_f32, 4.0, 5.0]; // second row values
    let clean = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &row_data).requires_grad_(true);
    let loss_clean = clean.cross_entropy(&targets);
    assert_tensor_eq(&loss, &loss_clean, 1e-5);
}

// BF16 round-trip (#7)
#[test]
fn test_bf16_roundtrip() {
    let original = Tensor::from_slice(DType::F32, Shape::new([3]), &[1.0f32, -2.5, 3.14]);
    let bf = original.to_dtype(DType::BF16);
    assert_eq!(bf.dtype, DType::BF16);
    let back = bf.to_dtype(DType::F32);
    let back_data = tensor_to_vec_f32(&back);
    assert!((back_data[0] - 1.0).abs() < 0.01);
    assert!((back_data[1] + 2.5).abs() < 0.1);
    assert!((back_data[2] - 3.14).abs() < 0.1);
}

// Gradient through device transfer (BUG)
#[test]
fn test_gradient_through_to() {
    // CPU leaf -> CPU (no-op) via .to(), which adds ContiguousOp.
    let leaf =
        Tensor::from_slice(DType::F32, Shape::new([3, 1]), &[1.0, 2.0, 3.0]).requires_grad_(true);
    let moved = leaf.to(Device::Cpu);
    let result = moved.sum();
    let grads = result.backward();
    let grad = grads.get(&leaf.id).expect("leaf should receive gradient");
    let grad_data = tensor_to_vec_f32(grad);
    assert_eq!(grad.shape.dims(), leaf.shape.dims());
    assert_eq!(grad_data, vec![1.0; 3]);
}

// ──────────────────────────────────────────────
// Autograd Numerical Gradient Checks (#8, #10)
// ──────────────────────────────────────────────
const EPS: f32 = 1e-3;
const TOL: f32 = 1e-2;

fn finite_diff<F>(f: F, input: &Tensor) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let mut grad = Tensor::zeros(input.dtype, input.shape.clone());
    let input_data = tensor_to_vec_f32(&input.ensure_cpu());
    let mut plus_data = input_data.clone();
    let mut minus_data = input_data.clone();

    for i in 0..input_data.len() {
        let orig = input_data[i];
        plus_data[i] = orig + EPS;
        let t_plus = Tensor::from_slice(input.dtype, input.shape.clone(), &plus_data);
        let out_plus = f(&t_plus).sum();

        minus_data[i] = orig - EPS;
        let t_minus = Tensor::from_slice(input.dtype, input.shape.clone(), &minus_data);
        let out_minus = f(&t_minus).sum();

        let diff =
            (tensor_to_vec_f32(&out_plus)[0] - tensor_to_vec_f32(&out_minus)[0]) / (2.0 * EPS);
        let mut grad_data = tensor_to_vec_f32(&grad);
        grad_data[i] = diff;
        grad = Tensor::from_slice(DType::F32, input.shape.clone(), &grad_data);

        plus_data[i] = orig;
        minus_data[i] = orig;
    }
    grad
}

fn check_gradients(op: impl Fn(&Tensor) -> Tensor, input_data: Vec<f32>, shape: Shape) {
    let input = Tensor::from_slice(DType::F32, shape.clone(), &input_data).requires_grad_(true);
    let output = op(&input).sum();
    let analytical_grads = output.backward();
    let analytical = analytical_grads.get(&input.id).unwrap();
    let numerical = finite_diff(op, &input);
    assert_tensor_eq(analytical, &numerical, TOL);
}

#[test]
fn test_relu_grad() {
    check_gradients(|x| x.relu(), vec![-1.0, 0.1, 0.5, 2.0], Shape::new([2, 2]));
}

#[test]
fn test_gelu_grad() {
    check_gradients(
        |x| x.gelu(),
        vec![-2.0, -0.5, 0.0, 0.5, 2.0],
        Shape::new([5]),
    );
}

#[test]
fn test_softmax_grad() {
    let data = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_slice(DType::F32, Shape::new([3]), &data).requires_grad_(true);
    let out = input.softmax().sum();
    let grads = out.backward();
    let analytical = grads.get(&input.id).unwrap();
    let numerical = finite_diff(|x| x.softmax().sum(), &input);
    assert_tensor_eq(analytical, &numerical, TOL);
}

#[test]
fn test_matmul_grad() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let a = Tensor::from_slice(DType::F32, Shape::new([2, 2]), &data_a).requires_grad_(true);
    let b = Tensor::from_slice(DType::F32, Shape::new([2, 2]), &data_b).requires_grad_(true);
    let c = a.matmul(&b).sum();
    let grads = c.backward();
    // Gradient w.r.t a
    let ga = grads.get(&a.id).unwrap();
    let num_a = finite_diff(
        |a| {
            let b_const = b.detach();
            a.matmul(&b_const).sum()
        },
        &a,
    );
    assert_tensor_eq(ga, &num_a, TOL);
    // Gradient w.r.t b
    let gb = grads.get(&b.id).unwrap();
    let num_b = finite_diff(
        |b| {
            let a_const = a.detach();
            a_const.matmul(&b).sum()
        },
        &b,
    );
    assert_tensor_eq(gb, &num_b, TOL);
}

#[test]
fn test_sum_mean_grad() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_slice(DType::F32, Shape::new([2, 2]), &data).requires_grad_(true);
    // sum
    let s = input.sum();
    let grads_s = s.backward();
    let gs = grads_s.get(&input.id).unwrap();
    assert_eq!(tensor_to_vec_f32(gs), vec![1.0; 4]);
    // mean
    let m = input.mean();
    let grads_m = m.backward();
    let gm = grads_m.get(&input.id).unwrap();
    assert_eq!(tensor_to_vec_f32(gm), vec![0.25; 4]);
}

// ──────────────────────────────────────────────
// Broadcasting (#4) & Ops
// ──────────────────────────────────────────────
#[test]
fn test_add_broadcast() {
    let a = Tensor::from_slice(
        DType::F32,
        Shape::new([2, 3]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let b = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[10.0, 20.0, 30.0]);
    let c = &a + &b;
    let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    assert_eq!(tensor_to_vec_f32(&c), expected);
}

#[test]
fn test_mul_broadcast_grad() {
    let a = Tensor::from_slice(DType::F32, Shape::new([2, 1]), &[2.0, 3.0]).requires_grad_(true);
    let b =
        Tensor::from_slice(DType::F32, Shape::new([1, 3]), &[4.0, 5.0, 6.0]).requires_grad_(true);
    let c = &a * &b;
    let s = c.sum();
    let grads = s.backward();
    let ga = grads.get(&a.id).unwrap();
    assert_eq!(tensor_to_vec_f32(ga), vec![15.0, 15.0]);
    let gb = grads.get(&b.id).unwrap();
    assert_eq!(tensor_to_vec_f32(gb), vec![5.0, 5.0, 5.0]);
}

// ──────────────────────────────────────────────
// View, Transpose, Cat, and other Tensor Engine methods
// ──────────────────────────────────────────────
#[test]
fn test_view_reshape() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::from_slice(DType::F32, Shape::new([2, 3]), &data);
    let v = t.view(Shape::new([3, 2]));
    assert_eq!(v.shape.dims(), [3, 2]);
    assert_eq!(tensor_to_vec_f32(&v), data);
}

#[test]
fn test_view_noncontiguous_triggers_copy() {
    let t = Tensor::from_slice(
        DType::F32,
        Shape::new([3, 4]),
        &(0..12).map(|x| x as f32).collect::<Vec<_>>(),
    );
    let narrow = t.narrow(1, 1, 2); // strides are no longer contiguous for columns
    assert!(!narrow.is_contiguous());
    let v = narrow.view(Shape::new([3, 2])); // should force contiguous copy
    assert!(v.is_contiguous());
    let expected = vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0];
    assert_eq!(tensor_to_vec_f32(&v), expected);
}

#[test]
fn test_cat_2d() {
    let a = Tensor::from_slice(
        DType::F32,
        Shape::new([2, 3]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let b = Tensor::from_slice(
        DType::F32,
        Shape::new([2, 3]),
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    let c = Tensor::cat(&[a, b]);
    assert_eq!(c.shape.dims(), [4, 3]);
    let expected: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    assert_eq!(tensor_to_vec_f32(&c), expected);
}

// ──────────────────────────────────────────────
// GPU consistency (placeholder)
// ──────────────────────────────────────────────
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_cpu_consistency() {
    // To be implemented with a real GpuContext
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_cpu_consistency() {
    // no-op when GPU not available
}

#[test]
fn debug_autograd_graph() {
    let input =
        Tensor::from_slice(DType::F32, Shape::new([3]), &[1.0, 2.0, 3.0]).requires_grad_(true);
    let out = input.relu().sum();
    let topo = out.topo_sort();
    eprintln!("Graph has {} tensors", topo.len());
    for t in &topo {
        eprintln!(
            "  id={:?} requires_grad={} has_node={}",
            t.id,
            t.requires_grad,
            t.node.is_some()
        );
    }
    let grads = out.backward();
    eprintln!("Gradient map keys: {:?}", grads.keys().collect::<Vec<_>>());
    assert!(grads.contains_key(&input.id), "Leaf gradient missing!");
}

#[test]
fn debug_broadcast_manual() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0f32, 20.0, 30.0];
    // Manually compute expected broadcast addition
    let expected: Vec<f32> = a_data
        .iter()
        .enumerate()
        .map(|(i, &x)| x + b_data[i % 3])
        .collect();
    // Now use the tensor operation
    let a = Tensor::from_slice(DType::F32, Shape::new([2, 3]), &a_data);
    let b = Tensor::from_slice(DType::F32, Shape::new([1, 3]), &b_data);
    let c = &a + &b;
    let c_vec = tensor_to_vec_f32(&c);
    eprintln!("Manual expected: {:?}", expected);
    eprintln!("Tensor result: {:?}", c_vec);
    assert_eq!(c_vec, expected);
}
