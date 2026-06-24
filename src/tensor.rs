use crate::device::Device;
use crate::gpu::GpuContext;
use crate::{CpuStorage, DType, Shape, Strides, autograd::Node};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use wgpu::util::DeviceExt;

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    pub fn next() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn raw(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    pub storage: Arc<CpuStorage>,
    pub byte_offset: usize,
    pub device: Device,

    // Autograd fields
    pub requires_grad: bool,

    // Box breaks the recursive size cycle!
    pub grad: Option<Box<Tensor>>,
    pub node: Option<Arc<Node>>,
}

// ... (rest of the file remains exactly the same) ...
impl Tensor {
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        let bytes = shape.num_elements() * dtype.size_in_bytes();
        Self {
            id: TensorId::next(),
            dtype,
            shape,
            strides,
            storage: Arc::new(CpuStorage::zeros(bytes)),
            device: crate::device::Device::Cpu,
            byte_offset: 0,
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    pub fn empty(dtype: DType, shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        let bytes = shape.num_elements() * dtype.size_in_bytes();
        Self {
            id: TensorId::next(),
            dtype,
            shape,
            strides,
            storage: Arc::new(CpuStorage::empty(bytes)),
            device: crate::device::Device::Cpu,
            byte_offset: 0,
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    pub fn from_slice<T: bytemuck::Pod>(dtype: DType, shape: Shape, data: &[T]) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let strides = shape.contiguous_strides();
        Self {
            id: TensorId::next(),
            dtype,
            shape,
            strides,
            storage: Arc::new(CpuStorage::from_bytes(bytes)),
            device: crate::device::Device::Cpu,
            byte_offset: 0,
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    /// Builder method to flag a tensor as a leaf variable that needs gradients (like Weights).
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        assert!(
            start + len <= self.shape.dims()[dim],
            "narrow out of bounds"
        );
        let mut new_shape = self.shape.clone();
        new_shape.dims_mut()[dim] = len;
        let stride = self.strides.steps()[dim];
        let byte_shift = (start as isize * stride) as usize * self.dtype.size_in_bytes();

        Self {
            id: TensorId::next(),
            dtype: self.dtype,
            shape: new_shape,
            strides: self.strides.clone(),
            storage: self.storage.clone(),
            byte_offset: self.byte_offset + byte_shift,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
            grad: None, // Views usually share gradients with their base tensor in a full impl
            node: None,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
    pub fn device(&self) -> &'static str {
        "cpu"
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        // Only use GPU if BOTH tensors are on GPU
        let out = if self.device.is_gpu() && other.device.is_gpu() {
            let ctx = match &self.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };
            GpuContext::add_gpu_resident(&ctx, self, other)
        } else {
            // For mixed or pure-CPU cases, ensure both are on CPU first
            let a_cpu = self.ensure_cpu();
            let b_cpu = other.ensure_cpu();
            crate::kernels::add(&a_cpu, &b_cpu)
        };

        if self.requires_grad || other.requires_grad {
            let op = std::sync::Arc::new(crate::ops::add::AddOp {
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
            });
            out.with_node(op, vec![self.clone(), other.clone()])
        } else {
            out
        }
    }

    /// Forward pass: Sum all elements to a scalar (Autograd aware)
    pub fn sum(&self) -> Tensor {
        let out = crate::kernels::sum_all(self);

        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::sum::SumOp {
                input_shape: self.shape.clone(),
                dtype: self.dtype,
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Reverse-mode Automatic Differentiation.
    /// Returns a HashMap mapping every leaf TensorId to its calculated Gradient.
    pub fn backward(&self) -> std::collections::HashMap<TensorId, Tensor> {
        let topo = self.topo_sort();
        let mut grads = std::collections::HashMap::new();

        // Seed the loss gradient with 1.0
        let seed = match self.dtype {
            DType::F32 => {
                Tensor::from_slice(DType::F32, Shape::new(Vec::<usize>::new()), &[1.0f32])
            }
            DType::F64 => {
                Tensor::from_slice(DType::F64, Shape::new(Vec::<usize>::new()), &[1.0f64])
            }
            _ => panic!("Unsupported dtype for backward"),
        };
        grads.insert(self.id, seed);

        // Traverse the graph from Loss -> Leaves
        for tensor in topo {
            if let Some(grad_out) = grads.get(&tensor.id) {
                if let Some(node) = &tensor.node {
                    let parent_grads = node.op.backward(grad_out);

                    // Distribute the gradients to the parents
                    for (parent, grad) in node.parents.iter().zip(parent_grads.into_iter()) {
                        if let Some(g) = grad {
                            // If a parent is used multiple times in the graph,
                            // its gradients must be accumulated (added together).
                            grads
                                .entry(parent.id)
                                .and_modify(|existing| {
                                    let summed = crate::kernels::add(existing, &g);
                                    *existing = summed;
                                })
                                .or_insert(g);
                        }
                    }
                }
            }
        }
        grads
    }

    /// Zero-cost transposition of a 2D tensor.
    /// We literally just swap the shapes and strides. No memory is copied!
    pub fn t(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "transpose only supports 2D tensors");
        Tensor {
            id: TensorId::next(),
            dtype: self.dtype,
            shape: self.shape.reverse(),
            strides: self.strides.reverse(),
            storage: self.storage.clone(), // Just clones the Arc!
            byte_offset: self.byte_offset,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
            grad: None,
            node: None,
        }
    }

    /// Forward pass: Matrix Multiplication (Autograd aware)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let out = if self.device.is_gpu() && other.device.is_gpu() {
            let ctx = match &self.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };
            GpuContext::matmul_gpu_resident(&ctx, self, other)
        } else {
            let a_cpu = self.ensure_cpu();
            let b_cpu = other.ensure_cpu();
            crate::kernels::matmul(&a_cpu, &b_cpu)
        };

        if self.requires_grad || other.requires_grad {
            let op = std::sync::Arc::new(crate::ops::matmul::MatMulOp {
                a: self.clone(),
                b: other.clone(),
            });
            out.with_node(op, vec![self.clone(), other.clone()])
        } else {
            out
        }
    }

    pub fn relu(&self) -> Tensor {
        let out = crate::kernels::activations::relu(self);
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::relu::ReluOp {
                input: self.clone(),
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Transfers tensor data between CPU and GPU.
    pub fn to(&self, device: Device) -> Tensor {
        if self.device == device {
            return self.clone(); // Already on the target device
        }

        match (&self.device, &device) {
            (Device::Cpu, Device::Gpu(gpu_ctx)) => {
                // CPU -> GPU: Upload data to VRAM
                let data = self.to_contiguous_bytes();
                let buffer = gpu_ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("gpu_tensor"),
                        contents: &data,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                Tensor {
                    id: TensorId::next(),
                    dtype: self.dtype,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    storage: Arc::new(CpuStorage::from_gpu_buffer(Arc::new(buffer))),
                    byte_offset: 0,
                    device: Device::Gpu(gpu_ctx.clone()),
                    requires_grad: self.requires_grad,
                    grad: None,
                    node: None,
                }
            }
            (Device::Gpu(_), Device::Cpu) => {
                // GPU -> CPU: Download data from VRAM
                let bytes = self.download_from_gpu();

                Tensor {
                    id: TensorId::next(),
                    dtype: self.dtype,
                    shape: self.shape.clone(),
                    strides: self.shape.contiguous_strides(),
                    storage: Arc::new(CpuStorage::from_bytes(&bytes)),
                    byte_offset: 0,
                    device: Device::Cpu,
                    requires_grad: self.requires_grad,
                    grad: None,
                    node: None,
                }
            }
            _ => panic!("Unsupported device transfer"),
        }
    }

    /// Materializes tensor data into a contiguous byte Vec (handles strides).
    fn to_contiguous_bytes(&self) -> Vec<u8> {
        if self.dtype != DType::F32 {
            panic!("Only F32 supported for now");
        }

        let num_elements = self.shape.num_elements();
        let mut result = Vec::with_capacity(num_elements * 4);

        let shape = self.shape.dims();
        let strides = self.strides.steps();
        let base = self.byte_offset / std::mem::size_of::<f32>();
        let ptr = self.storage.as_ptr() as *const f32;

        for i in 0..num_elements {
            let mut offset = 0isize;
            let mut idx = i;
            for d in (0..shape.len()).rev() {
                let dim_size = shape[d];
                let coord = idx % dim_size;
                idx /= dim_size;
                offset += coord as isize * strides[d];
            }
            let val = unsafe { *ptr.add(offset as usize + base) };
            result.extend_from_slice(&val.to_le_bytes());
        }

        result
    }

    /// Downloads GPU buffer data back to CPU.
    fn download_from_gpu(&self) -> Vec<u8> {
        let gpu_ctx = match &self.device {
            Device::Gpu(ctx) => ctx,
            _ => panic!("Not a GPU tensor"),
        };

        let buffer = self.storage.get_gpu_buffer().unwrap();
        let bytes = self.shape.num_elements() * self.dtype.size_in_bytes();

        // Create staging buffer
        let staging = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU -> staging
        let mut encoder = gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes as wgpu::BufferAddress);
        gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        gpu_ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging.unmap();

        result
    }

    /// Ensures tensor data is accessible on CPU.
    /// If already on CPU, returns self. If on GPU, downloads and returns new CPU tensor.
    /// This is a temporary bridge for CPU-only kernels operating on GPU tensors.
    pub fn ensure_cpu(&self) -> Tensor {
        if self.device.is_cpu() {
            self.clone()
        } else {
            self.to(Device::Cpu)
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let out = crate::kernels::scalar_mul(self, scalar);
        if self.requires_grad {
            // The derivative of (x * c) is just (grad * c)
            let op = std::sync::Arc::new(crate::ops::scalar_mul::ScalarMulOp { scalar });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    pub fn softmax(&self) -> Tensor {
        let out = crate::kernels::softmax(self);
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::softmax::SoftmaxOp {
                output: out.clone(),
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides()
    }

    /// Forces a non-contiguous tensor to become contiguous in memory by copying data.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        let out = Tensor::empty(self.dtype, self.shape.clone());
        crate::kernels::copy(self, &out);
        out
    }

    /// Zero-cost reshaping. If the tensor is non-contiguous, it forces a copy first.
    pub fn view(&self, new_shape: Shape) -> Tensor {
        assert_eq!(
            self.shape.num_elements(),
            new_shape.num_elements(),
            "Cannot view: element count mismatch"
        );

        let base = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };

        let out = Tensor {
            id: TensorId::next(),
            dtype: base.dtype,
            shape: new_shape.clone(),
            strides: new_shape.contiguous_strides(),
            storage: base.storage,
            byte_offset: base.byte_offset,
            device: base.device,
            requires_grad: base.requires_grad,
            grad: None,
            node: None,
        };

        if base.requires_grad {
            let op = std::sync::Arc::new(crate::ops::view::ViewOp {
                original_shape: self.shape.clone(),
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Zero-cost transposition. Just swaps shapes and strides!
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Tensor {
        let out = Tensor {
            id: TensorId::next(),
            dtype: self.dtype,
            shape: self.shape.transpose(dim1, dim2),
            strides: self.strides.transpose(dim1, dim2),
            storage: self.storage.clone(),
            byte_offset: self.byte_offset,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            node: None,
        };

        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::transpose::TransposeOp { dim1, dim2 });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Systems trick for Batched MatMul: Extracts a 2D slice from a 3D tensor [Batch, M, N]
    /// without copying memory, by just shifting the byte_offset!
    pub fn get_2d_slice(&self, batch_idx: usize) -> Tensor {
        assert_eq!(self.rank(), 3);
        let m = self.shape.dims()[1];
        let n = self.shape.dims()[2];
        let stride_0 = self.strides.steps()[0];
        let byte_shift = (batch_idx as isize * stride_0) as usize * self.dtype.size_in_bytes();

        Tensor {
            id: TensorId::next(),
            dtype: self.dtype,
            shape: Shape::new([m, n]),
            strides: Shape::new([m, n]).contiguous_strides(),
            storage: self.storage.clone(),
            byte_offset: self.byte_offset + byte_shift,
            device: self.device.clone(),
            requires_grad: false, // Internal view for manual batching
            grad: None,
            node: None,
        }
    }

    /// Concatenates a list of 2D tensors along a new leading dimension (dim 0).
    /// Fully integrated into the Autograd graph.
    pub fn cat(tensors: &[Tensor]) -> Tensor {
        let out = crate::kernels::cat_2d(tensors);
        let requires_grad = tensors.iter().any(|t| t.requires_grad);

        if requires_grad {
            let op = std::sync::Arc::new(crate::ops::cat::CatOp {
                n: tensors.len(),
                s: tensors[0].shape.dims()[0],
                d: tensors[0].shape.dims()[1],
            });
            out.with_node(op, tensors.to_vec())
        } else {
            out
        }
    }
}
