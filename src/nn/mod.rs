// --------------------------------------------------------------------------
// Module: nn
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Defines the composable architecture subsystem for Haelixe.
//   This module introduces the `Module` trait, which serves as the
//   foundational contract for all neural network layers, allowing them
//   to be stacked, nested, and queried for their trainable parameters.
//
// HISTORICAL CONTEXT:
//   Forged in July 2026 to replace the manual parameter tracking
//   vectors used in the Integration Crucible. This abstraction is
//   mandatory for scaling the engine to multi-layer Transformers.
//
// CALL GRAPH:
//   Called by: Downstream consumer models (e.g., TransformerBlock).
//   Calls: Tensor Engine operations.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

pub mod linear;
pub mod rms_norm;
pub use linear::Linear;
pub use rms_norm::RMSNorm;

use crate::Tensor;

/// The foundational contract for all composable neural network layers.
///
/// Any struct implementing this trait can be nested inside other layers.
/// The `parameters()` method must recursively collect all leaf tensors
/// that have `requires_grad` set to true, ensuring the Optimizer can
/// seamlessly discover and update the entire model's state.
pub trait Module {
    /// Executes the forward pass of the layer.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Recursively extracts all trainable master weights.
    fn parameters(&self) -> Vec<&Tensor>;
}
