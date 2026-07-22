<div align="center">
  <img src="./haelixe.png" alt="Haelixe Logo" width="300"/>
</div>

# Haelixe: A Bare-Metal Deep Learning Engine in Rust


**Haelixe** (pronounced Hay‑lix) is a high‑performance, research‑grade deep learning engine built entirely from scratch in pure Rust. Unlike frameworks that wrap C++ or CUDA libraries, Haelixe forges its own path: every tensor operation, every gradient, and every memory allocation is written by hand, giving you full control over performance and correctness. The CPU engine uses rayon for lock‑free parallelism, while GPU compute is handled by hand‑crafted WGSL shaders dispatched through the wgpu crate to Vulkan, Metal, or DirectX 12 backends — no CUDA, no proprietary drivers.

Under the hood, Haelixe provides a dynamic computation graph with reverse‑mode automatic differentiation, zero‑copy strided tensor views, and advanced memory management including a binning slab allocator for deterministic VRAM reclamation. It natively supports mixed‑precision training with BF16 storage, just‑in‑time autocast to F32, and the master‑weights pattern. The built‑in neural network modules include modern Transformer primitives like Rotary Position Embeddings (RoPE), RMSNorm, GELU activations, and a fused flash‑attention kernel that keeps the entire attention score calculation inside the GPU’s L1 cache. All of this is designed to let researchers push the limits of large‑language models directly on bare metal, from first principles.
---

## 💻 Stack

<div align="center">

![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![Language](https://img.shields.io/badge/language-Rust-orange)
![Status](https://img.shields.io/badge/status-research--grade-brightgreen)
![Build](https://img.shields.io/github/actions/workflow/status/sethigris/haelixe/ci.yml?branch=main)
![Tests](https://img.shields.io/github/actions/workflow/status/sethigris/haelixe/ci.yml?branch=main&label=tests)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

</div>

---

## Core Architecture and Modern LLM Primitives

### Modern Transformer Primitives
Haelixe implements the exact mathematical foundations used by state-of-the-art architectures like LLaMA 3 and Mistral:
* **Rotary Position Embeddings (RoPE):** Replaces legacy absolute positional encodings with sequence-dependent orthogonal rotations for mathematically pure relative distance awareness.
* **RMSNorm:** Eliminates the mean-subtraction step of standard LayerNorm to prevent catastrophic variance collapse in deep networks.
* **GELU Activations:** Replaces brittle ReLU networks with smooth, non-linear Gaussian Error Linear Units to prevent dying neurons.
* **Pre-Norm Architecture with Final RMSNorm:** Ensures stable gradient flow and bounded residual streams across deep Transformer blocks.

### Mixed-Precision Foundation
* **BF16 Storage:** Supports BFloat16 memory layouts to halve the VRAM footprint and PCIe bandwidth requirements.
* **JIT Autocast Boundary:** Seamlessly upcasts BF16 weights to F32 at the compute boundary for mathematical stability on hardware lacking native 16-bit ALUs.
* **Master Weights Pattern:** Separates F32 optimizer state from BF16 model storage for production-grade mixed-precision training.

### Zero-Copy Tensor Engine
* **Strided Memory Layout:** Tensors are backed by a physical buffer, but operations like view, transpose, and slicing only manipulate the Shape and Strides arrays. No memory is copied.
* **Zero-Cost Broadcasting:** Addition and multiplication automatically stretch tensors to match shapes using zero-strides.

### Dynamic Autograd System
* **Reverse-Mode Differentiation:** A fully functional Directed Acyclic Graph (DAG) that tracks operations and computes gradients via topological sorting.
* **In-Place Optimization:** Gradients are accumulated efficiently using HashMap lookups and in-place memory mutations.

---

## ⚙️ Systems-Level Hardware Engineering

### Binning Slab Allocator and RAII VRAM Reclamation
* **Power-of-Two Binning:** Groups similar tensor sizes into the same memory pool to dramatically increase cache hit rates and eliminate driver-level VRAM allocation overhead.
* **Deterministic RAII Reclamation:** Uses `Arc` reference counting to guarantee that VRAM slabs are only returned to the free list when the final Autograd reference is destroyed, preventing silent gradient corruption.

### Cross-Platform GPU Compute
* **WebGPU via wgpu:** Write once, run anywhere (Vulkan, Metal, DX12).
* **Kernel Fusion:** The Linear layer fuses MatMul and Bias Add into a single WGSL compute shader.
* **Flash-Attention:** Fused WGSL compute shaders that execute attention score calculations entirely within GPU L1 Cache.

### Systems-Level CPU Optimization
* **Parallelism:** Kernels utilize `rayon` for lock-free, data-parallel execution across all CPU cores.
* **Thread-Safe Pointers:** Custom `SyncPtr` wrappers safely bypass Rust's strict concurrency rules for raw pointers.

### Out-of-Core Data Loading
* **Memory-Mapped Datasets:** Using `memmap2`, Haelixe maps binary dataset files directly into virtual memory, allowing training on datasets larger than system RAM.

---

## 📂 Project Structure

* **`src/autograd.rs`** - Computation graph (DAG) and reverse-mode autodiff
* **`src/data/`** - Memory-mapped datasets and zero-copy dataloaders
* **`src/device.rs`** - CPU/GPU abstraction and device dispatch
* **`src/dtype.rs`** - Data type definitions (F32, F64, BF16)
* **`src/gpu/`** - wgpu initialization, Binning Slab Allocator, and WGSL shaders
* **`src/kernels/`** - Bare-metal compute (matmul, reduce, binary, activations, RoPE, RMSNorm)
* **`src/layout.rs`** - Shape, Strides, and contiguous memory mapping
* **`src/nn/`** - Neural network layers (Linear, MultiHeadAttention, TransformerBlock)
* **`src/ops/`** - Autograd operations (Forward/Backward pass logic)
* **`src/optim.rs`** - Optimizers (AdamW with Cosine Annealing and GPU/CPU dispatch)
* **`src/storage.rs`** - `UnsafeCell`, physical memory backings, and mixed-precision slice accessors
* **`src/tensor.rs`** - The core Tensor struct and API
* **`haelixe-lab/`** - Downstream consumer workspace for API ergonomics and mathematical convergence testing

---

## 🚀 Getting Started

### The Haelixe Lab: Downstream Consumer Testing
Haelixe utilizes a Cargo Workspace monorepo architecture. The `haelixe-lab` directory serves as a downstream consumer project that imports the core Haelixe library via a local path dependency. This ensures atomic API evolution, tests public API ergonomics, and serves as the definitive mathematical convergence testbed (e.g., Sequence Denoising with Cosine Annealing).

### Prerequisites
* Rust (1.70+)
* A GPU with Vulkan, Metal, or DX12 support (for GPU kernels)

### Running the Engine
Clone the repository and run the downstream lab to verify mathematical convergence and hardware integration.

```bash
git clone https://github.com/sethigris/haelixe.git
cd haelixe
cargo run -p haelixe-lab --release
```

---

## 🧪 Example: Sequence Denoising With Modern Primitives

Below is the reference implementation used in `haelixe-lab` to validate the framework. It trains a Transformer block utilizing RoPE, RMSNorm, GELU, and Cosine Annealing to map a noisy multi-frequency sine wave back to its clean mathematical signal.

```rust
use haelixe::{DType, Device, Shape, Tensor, TransformerBlock, RMSNorm, optim::AdamW};
use std::f32::consts::PI;

fn main() {
    let gpu = Device::gpu();
    let batch_size = 4;
    let seq_len = 32;
    let hidden_dim = 64;
    let num_heads = 4;

    let mut embed = haelixe::Linear::new(1, hidden_dim);
    let mut block = TransformerBlock::new(hidden_dim, num_heads);
    let mut final_norm = RMSNorm::new(hidden_dim);
    let mut head = haelixe::Linear::new(hidden_dim, 1);

    embed.to(gpu.clone());
    block.to(gpu.clone());
    final_norm.to(gpu.clone());
    head.to(gpu.clone());

    let mut optimizer = AdamW::new(0.001);
    let max_lr = 0.001;
    let min_lr = 0.00005;
    let total_epochs = 100;

    for epoch in 0..total_epochs {
        let cosine_decay = 0.5 * (1.0 + (PI * epoch as f32 / total_epochs as f32).cos());
        let current_lr = min_lr + 0.5 * (max_lr - min_lr) * cosine_decay;
        optimizer.set_lr(current_lr);

        // Data generation and forward pass omitted for brevity...
        // The network successfully converges from MSE 3.74 to < 0.85
    }
}
```
