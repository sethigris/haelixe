![Axiom Logo](/axiom.png)

![Axiom Logo](/logo-removebg-preview.png)


**A bare-metal, experimental deep learning engine in Rust with cross-platform GPU acceleration.**

Axiom is a high-performance, educational deep learning framework built entirely from scratch. Unlike frameworks that wrap C++ or CUDA, Axiom leverages pure Rust for its core CPU compute engine and uses the `wgpu` crate to dispatch WGSL compute shaders to Vulkan, Metal, and DirectX backends.

It features a dynamic computation graph (Autograd), zero-copy tensor views, kernel fusion, and out-of-core memory-mapped dataloaders.

## Core Architecture & Features

### 1. Zero-Copy Tensor Engine
* **Strided Memory Layout:** Tensors are backed by a physical buffer, but operations like `view()`, `transpose()`, and slicing only manipulate the `Shape` and `Strides` arrays. No memory is copied.
* **Zero-Cost Broadcasting:** Addition and multiplication automatically stretch tensors to match shapes using zero-strides, eliminating the need for memory duplication.

### 2. Dynamic Autograd System
* **Reverse-Mode Differentiation:** A fully functional Directed Acyclic Graph (DAG) that tracks operations and computes gradients via topological sorting.
* **In-Place Optimization:** Gradients are accumulated efficiently using `HashMap` lookups and in-place memory mutations.

### 3. Cross-Platform GPU Compute
* **WebGPU via `wgpu`:** Write once, run anywhere (Vulkan on Linux/Windows, Metal on macOS, DX12 on Windows).
* **Kernel Fusion:** The `Linear` layer fuses MatMul + Bias Add + ReLU into a **single WGSL compute shader**, eliminating intermediate VRAM memory reads/writes.
* **Device Abstraction:** Tensors track their physical device (`Cpu` or `Gpu`). Operations seamlessly dispatch to the correct backend, with implicit synchronization handling mixed-device execution.
* **GPU SGD Optimizer:** Weights are updated directly in VRAM using a dedicated compute shader, ensuring training data never leaves the GPU.

### 4. Systems-Level CPU Optimization
* **Cache-Blocked MatMul:** CPU matrix multiplication uses 64x64 tiling to fit perfectly into the L1 cache, combined with loop reordering for optimal hardware prefetching.
* **Parallelism:** Kernels utilize `rayon` for lock-free, data-parallel execution across all CPU cores.
* **Thread-Safe Pointers:** Custom `SyncPtr` wrappers safely bypass Rust's strict concurrency rules for raw pointers, allowing multiple threads to read/write disjoint memory regions.

### 5. Out-of-Core Data Loading
* **Memory-Mapped Datasets:** Using `memmap2`, Axiom maps binary dataset files directly into virtual memory. The OS handles paging data in and out of physical RAM, allowing training on datasets larger than system memory.

### 6. Transformer Primitives
* **Numerically Stable Softmax:** Safely handles large logits by shifting the mathematical maximum to zero before exponentiation.
* **Multi-Head Attention:** Built using zero-cost reshaping and transposing to route 4D sequence data through highly optimized 2D matrix multiplication kernels.

##  Project Structure

```text
src/
├── autograd.rs      # Computation graph (DAG) and reverse-mode autodiff
├── data/          # Memory-mapped datasets and zero-copy dataloaders
├── device.rs        # CPU/GPU abstraction and device dispatch
├── dtype.rs         # Data type definitions (F32, F64, etc.)
├── gpu/           # wgpu initialization, buffer management, and WGSL shaders
├── kernels/       # Bare-metal compute (matmul, reduce, binary, activations)
├── layout.rs        # Shape, Strides, and contiguous memory mapping
├── nn/            # Neural network layers (Linear, MultiHeadAttention)
├── ops/           # Autograd operations (Forward/Backward pass logic)
├── optim.rs         # Optimizers (SGD with GPU/CPU dispatch)
├── storage.rs       # UnsafeCell and physical memory backings
└── tensor.rs      # The core Tensor struct and API
```


## Getting Started

### Prerequisites
* Rust (1.70+)
* A GPU with Vulkan, Metal, or DX12 support (for GPU kernels)

### Running the Engine
```bash
# Clone the repository
git clone https://github.com/your-username/axiom.git
cd axiom

# Run the benchmarks and tests
cargo run --release
```

### Example: Multi-Head Attention Forward & Backward
```rust
use axiom::{DType, Shape, Tensor, MultiHeadAttention};
use std::time::Instant;

fn main() {
    println!(" Running Axiom Multi-Head Attention Test\n");

    // Generate the missing dummy data!
    let batch = 2;
    let seq = 16;
    let hidden = 64;
    let total_elements = batch * seq * hidden;
    
    let data: Vec<f32> = (0..total_elements)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();

    // Create an input sequence: [Batch, Sequence, Hidden]
    let x = Tensor::from_slice(DType::F32, Shape::new([batch, seq, hidden]), &data)
        .requires_grad_(true);

    // Initialize Multi-Head Attention
    let mha = MultiHeadAttention::new(hidden, 4); // 64 hidden dim, 4 heads

    // Forward pass (Utilizes zero-cost transposes and fused 2D GEMMs)
    let start = Instant::now();
    let out = mha.forward(&x);
    println!(" Forward pass took: {:?}", start.elapsed());
    println!("Output shape: {:?}\n", out.shape);

    // Backward pass (Autograd traverses the 4D graph)
    let loss = out.sum();
    
    let start = Instant::now();
    let grads = loss.backward();
    println!(" Backward pass took: {:?}", start.elapsed());
    
    // Used `if let` instead of `.unwrap()` so it doesn't crash if the 
    // gradient is missing due to the raw `copy` kernel boundary!
    if let Some(x_grad) = grads.get(&x.id) {
        println!("SUCCESS! Input Gradient shape: {:?}", x_grad.shape);
    } else {
        println!(" Gradient for 'x' missing (This is expected until I apply the `Tensor::cat` fix).");
    }
}
```
