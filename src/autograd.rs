use crate::{Tensor, TensorId};
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;

// --------------------------------------------------------------------------
// MODULE: autograd (State Management Additions)
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Provides global state management for gradient tracking.
//   Introduces the `NoGradGuard` RAII pattern to temporarily disable
//   the construction of the computational graph during inference or
//   validation phases, preventing catastrophic VRAM/RAM leaks.
//
// STATE TRANSITION DIAGRAM:
//   [Grad Enabled] --(NoGradGuard::new)--> [Grad Disabled]
//   [Grad Disabled] --(Guard Dropped)--> [Grad Enabled]
// --------------------------------------------------------------------------

thread_local! {
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

/// Returns true if gradient tracking is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

/// Sets the global gradient state.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|c| c.set(enabled));
}

/// A RAII guard that disables gradient tracking for the current scope.
/// When dropped, it restores the previous state, allowing safe nesting.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

/// The core trait for all neural network operations.
/// Added `std::fmt::Debug` so the Node can derive Debug.
pub trait Op: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    /// Given the gradient of the output, calculate the gradients of the inputs.
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
}

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node {
    pub op: Arc<dyn Op>,
    pub parents: Vec<Tensor>,
}

impl Tensor {
    /// Attaches a computation node to this tensor, marking it as part of the Autograd graph.
    pub fn with_node(mut self, op: Arc<dyn Op>, parents: Vec<Tensor>) -> Self {
        // Bypass graph construction if globally disabled
        if !is_grad_enabled() {
            return self;
        }

        self.node = Some(Arc::new(Node { op, parents }));
        self.requires_grad = true; // If it's an output of an Op, it requires grad
        self
    }

    /// Performs a Depth-First Search (DFS) to build a reverse topological sort of the graph.
    /// This is the exact order required for backpropagation (loss -> leaves).
    pub fn topo_sort(&self) -> Vec<Tensor> {
        let mut visited = HashSet::new();
        let mut sorted = Vec::new();

        fn dfs(tensor: &Tensor, visited: &mut HashSet<TensorId>, sorted: &mut Vec<Tensor>) {
            if visited.contains(&tensor.id) {
                return;
            }
            visited.insert(tensor.id);

            // If this tensor was created by an operation, visit its parents first
            if let Some(node) = &tensor.node {
                for parent in &node.parents {
                    dfs(parent, visited, sorted);
                }
            }

            // Post-order traversal: parents are added to the list before the child
            sorted.push(tensor.clone());
        }

        dfs(self, &mut visited, &mut sorted);

        // Post-order gives us Forward order (Parents -> Children).
        // We reverse it to get Backward order (Children -> Parents / Loss -> Leaves).
        sorted.reverse();
        sorted
    }
}
