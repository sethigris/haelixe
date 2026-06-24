/// The logical dimensions of a tensor (e.g., [Batch, Channels, Height, Width]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

/// Memory offset steps for each dimension.
/// We use `isize` instead of `usize` to support negative strides,
/// which allows zero-cost tensor flipping/reversing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<isize>);

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().collect())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        if self.0.is_empty() {
            return 1; // Scalar tensor
        }
        self.0.iter().product()
    }

    /// Calculates the row-major (C-style) contiguous strides for this shape.
    pub fn contiguous_strides(&self) -> Strides {
        let mut strides = vec![0isize; self.rank()];
        let mut current_stride = 1isize;

        // Row-major means the last dimension is contiguous in memory (stride = 1).
        // We iterate backwards to build this up correctly.
        for i in (0..self.rank()).rev() {
            strides[i] = current_stride;
            current_stride *= self.0[i] as isize;
        }

        Strides(strides)
    }

    pub fn dims_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    /// Computes the resulting shape when broadcasting `self` and `other`.
    /// Returns `None` if the shapes are incompatible (e.g., [3] and [4]).
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let rank_a = self.rank();
        let rank_b = other.rank();
        let max_rank = rank_a.max(rank_b);
        let mut result = Vec::with_capacity(max_rank);

        // Iterate from innermost (rightmost) dimension backwards
        for i in 0..max_rank {
            let dim_a = if i < rank_a {
                self.0[rank_a - 1 - i]
            } else {
                1
            };
            let dim_b = if i < rank_b {
                other.0[rank_b - 1 - i]
            } else {
                1
            };

            if dim_a == dim_b {
                result.push(dim_a);
            } else if dim_a == 1 {
                result.push(dim_b);
            } else if dim_b == 1 {
                result.push(dim_a);
            } else {
                return None; // Incompatible shapes
            }
        }
        result.reverse();
        Some(Shape(result))
    }

    /// Reverses the dimensions (e.g., [M, N] -> [N, M]).
    /// Used for zero-cost transposition.
    pub fn reverse(&self) -> Self {
        let mut dims = self.0.clone();
        dims.reverse();
        Self(dims)
    }

    /// Swaps two dimensions (e.g., turning [Batch, Seq, Heads, Dim] into [Batch, Heads, Seq, Dim])
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut data = self.0.clone();
        data.swap(dim1, dim2);
        Self(data)
    }
}

impl Strides {
    pub fn new(steps: impl IntoIterator<Item = isize>) -> Self {
        Self(steps.into_iter().collect())
    }

    pub fn steps(&self) -> &[isize] {
        &self.0
    }

    /// Calculates the strides required to broadcast `shape` to `target_shape`.
    pub fn broadcast_to(shape: &Shape, strides: &Strides, target_shape: &Shape) -> Strides {
        let mut result = vec![0isize; target_shape.rank()];
        let shape_dims = shape.dims();
        let stride_steps = strides.steps();

        let target_rank = target_shape.rank();
        let shape_rank = shape_dims.len();

        for i in 0..target_rank {
            let target_dim = target_shape.dims()[target_rank - 1 - i];
            let shape_idx = shape_rank as isize - 1 - i as isize;

            if shape_idx >= 0 {
                let shape_dim = shape_dims[shape_idx as usize];
                let original_stride = stride_steps[shape_idx as usize];

                if shape_dim == target_dim {
                    // Dimensions match, keep original stride
                    result[target_rank - 1 - i] = original_stride;
                } else if shape_dim == 1 {
                    // Dimension is 1, set stride to 0 to repeat the value
                    result[target_rank - 1 - i] = 0;
                } else {
                    panic!(
                        "Invalid broadcast: shape dim {} cannot be broadcast to target dim {}",
                        shape_dim, target_dim
                    );
                }
            } else {
                // The shape has been virtually prepended with 1s. Stride is 0.
                result[target_rank - 1 - i] = 0;
            }
        }

        Strides(result)
    }

    /// Reverses the strides to match the reversed shape.
    pub fn reverse(&self) -> Self {
        let mut steps = self.0.clone();
        steps.reverse();
        Self(steps)
    }

    /// Swaps two dimensions (e.g., turning [Batch, Seq, Heads, Dim] into [Batch, Heads, Seq, Dim])
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut data = self.0.clone();
        data.swap(dim1, dim2);
        Self(data)
    }
}
