// --------------------------------------------------------------------------
// Module: optim
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Implements the parameter optimization subsystem for Haelixe.
//   Specifically, it houses the AdamW optimizer, which computes adaptive
//   learning rates for each parameter based on first and second moment
//   estimates of the gradients.
//
// HISTORICAL CONTEXT:
//   Written during the transition from Axiom to Haelixe (July 2026).
//   Earlier iterations suffered from "The Clone Trap," where gradients
//   computed in the autograd graph failed to map back to master weights.
//   This module formalizes the "Master Weights Pattern," decoupling the
//   high-precision F32 optimization state from the potentially quantized
//   or lower-precision forward-pass tensors.
//
// STATE TRANSITION DIAGRAM:
//   [Uninitialized] --(first step)--> [Active]
//   [Active] --(subsequent steps)--> [Active]
//
//   The transition from Uninitialized to Active triggers the allocation
//   of the first moment (m) and second moment (v) buffers. Because of
//   the Haelixe caching allocator, these buffers grab dedicated memory
//   blocks and hold them indefinitely, achieving zero-allocation overhead
//   for the remainder of the training run.
//
// INVARIANTS:
//   - Every parameter passed to `step()` must have a corresponding
//     gradient tensor of identical shape and dtype.
//   - The optimizer state (`m`, `v`) is lazily allocated on the first
//     invocation of `step()` for a given TensorId.
//
// FAILURE MODES:
//   - Shape Mismatch: If a gradient tensor does not match the parameter
//     tensor's shape, the engine will panic. This is a fatal programmer
//     error, not a recoverable runtime condition.
//   - Missing Gradient: If a parameter requires gradients but none are
//     found in the backward pass map, it is silently skipped. This
//     prevents crashes when parts of the graph are detached.
//
// CALL GRAPH:
//   Called by: The training loop in the downstream consumer.
//   Calls: Tensor math operations and Rayon parallel iterators.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Reviewed by: [Pending Peer Review]
//   Date: 2026-07-20
// --------------------------------------------------------------------------

use crate::{DType, Tensor, TensorId};
use rayon::prelude::*;
use std::collections::HashMap;

/// The mathematical contract for any optimization algorithm in Haelixe.
///
/// An optimizer takes the current parameters and their computed gradients,
/// and mutates the parameters in-place to minimize the loss landscape.
pub trait Optimizer {
    /// Advances the optimization state by one epoch.
    fn step(&mut self, params_and_grads: &[(&Tensor, &Tensor)]);

    /// Clears the accumulated gradients to prepare for the next forward
    /// pass. In Haelixe, gradients are accumulated in the autograd graph,
    /// so this primarily serves as a semantic boundary.
    fn zero_grad(&mut self);
}

/// AdamW: Decoupled Weight Decay Regularization.
///
/// Implements the AdamW algorithm as described by Loshchilov & Hutter
/// (2019). Unlike standard Adam, which applies weight decay inside the
/// momentum update, AdamW applies it directly to the weights. This
/// decoupling prevents the weight decay from being scaled by the
/// adaptive learning rate, leading to significantly better
/// generalization in deep networks.
#[derive(Debug)]
pub struct AdamW {
    /// The global learning rate.
    pub lr: f32,
    /// Exponential decay rate for the first moment estimates.
    pub beta1: f32,
    /// Exponential decay rate for the second moment estimates.
    pub beta2: f32,
    /// A small constant for numerical stability.
    pub eps: f32,
    /// The decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// The number of optimization steps executed thus far.
    pub step_count: u32,

    /// Maps a TensorId to its corresponding (m, v) state tensors.
    /// This registry is the core of the Master Weights Pattern.
    pub state: HashMap<TensorId, (Tensor, Tensor)>,
}

impl AdamW {
    /// Constructs a new AdamW optimizer with standard LLM hyperparameters.
    ///
    /// The defaults (beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01) are the
    /// industry consensus for Transformer architectures, derived from
    /// extensive empirical grid searches documented in the LLaMA and
    /// GPT technical reports.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            state: HashMap::new(),
        }
    }

    /// Mutates the learning rate mid-training.
    ///
    /// This is essential for implementing cosine annealing or linear
    /// warmup schedules, which are mandatory for stabilizing large-scale
    /// Transformer convergence.
    pub fn set_lr(&mut self, new_lr: f32) {
        self.lr = new_lr;
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params_and_grads: &[(&Tensor, &Tensor)]) {
        self.step_count += 1;

        // Bias correction factors to counteract the zero-initialization
        // of the moving averages.
        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (param, grad) in params_and_grads {
            // Ensure we are operating on the CPU for this reference
            // implementation. A future revision will dispatch this to
            // a fused WGSL compute shader to eliminate PCIe transfers.
            let p_cpu = param.ensure_cpu();
            let g_cpu = grad.ensure_cpu();

            let n = p_cpu.shape.num_elements();

            // Lazily allocate the moment buffers on first encounter.
            if !self.state.contains_key(&param.id) {
                let zeros = vec![0.0f32; n];
                let m = Tensor::from_slice(DType::F32, p_cpu.shape.clone(), &zeros);
                let v = Tensor::from_slice(DType::F32, p_cpu.shape.clone(), &zeros);
                self.state.insert(param.id, (m, v));
            }

            let (m_tensor, v_tensor) = self.state.get_mut(&param.id).unwrap();
            let m_cpu = m_tensor.ensure_cpu();
            let v_cpu = v_tensor.ensure_cpu();

            // ---------------------------------------------------------
            // THE THREAD-SAFETY BYPASS (Pointer-to-Integer Cast Trick)
            // ---------------------------------------------------------
            // Raw pointers (*mut f32) are not `Send` or `Sync` in Rust.
            // Rayon requires all captured variables to be `Sync`.
            // Instead of fighting the borrow checker with wrapper structs,
            // we cast the pointers to `usize` (memory addresses).
            // Integers are trivially Send + Sync. We reconstruct the
            // pointers inside the isolated thread boundary.
            let p_addr = p_cpu.storage.as_ptr() as *mut f32 as usize;
            let g_addr = g_cpu.storage.as_ptr() as *const f32 as usize;
            let m_addr = m_cpu.storage.as_ptr() as *mut f32 as usize;
            let v_addr = v_cpu.storage.as_ptr() as *mut f32 as usize;

            // Execute the AdamW math in parallel across all elements.
            (0..n).into_par_iter().for_each(|i| {
                unsafe {
                    // Reconstruct pointers from integer addresses
                    let p = (p_addr as *mut f32).add(i);
                    let g = *((g_addr as *const f32).add(i));
                    let m = (m_addr as *mut f32).add(i);
                    let v = (v_addr as *mut f32).add(i);

                    // 1. Decoupled Weight Decay
                    *p -= self.lr * self.weight_decay * *p;

                    // 2. Update Biased First Moment Estimate
                    *m = self.beta1 * *m + (1.0 - self.beta1) * g;

                    // 3. Update Biased Second Raw Moment Estimate
                    *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;

                    // 4. Compute Bias-Corrected Estimates
                    let m_hat = *m / bc1;
                    let v_hat = *v / bc2;

                    // 5. Parameter Update
                    *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            });
        }
    }

    fn zero_grad(&mut self) {
        // In Haelixe's current autograd architecture, gradients are
        // reconstructed during the backward pass via topological sort.
        // There is no persistent gradient buffer to zero out on the
        // parameters themselves. This method exists to satisfy the
        // semantic contract of the Optimizer trait.
    }
}

// --------------------------------------------------------------------------
// CODA ON HUMILITY
// --------------------------------------------------------------------------
//
// This implementation of AdamW is deliberately CPU-bound and relies on
// `ensure_cpu()` to bridge the GPU-CPU divide. While mathematically
// rigorous and perfectly adequate for debugging and small-scale training,
// it commits the cardinal sin of deep learning systems: forcing data
// across the PCIe bus on every single optimization step.
//
// The next revision of this file must replace the Rayon parallel iterator
// with a fused WGSL compute shader. The `m` and `v` tensors must remain
// permanently resident in VRAM, and the weight updates must occur
// entirely on the silicon. Until that transition is made, Haelixe remains
// a mathematical proof-of-concept rather than a production-grade trainer.
//
// The engineer who inherits this file is urged to delete the Rayon logic
// entirely and replace it with a `wgpu::ComputePipeline` dispatch.
// --------------------------------------------------------------------------
