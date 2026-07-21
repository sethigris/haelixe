use crate::device::Device;
use crate::gpu::GpuContext;
use crate::{CpuStorage, DType, Shape, Strides, autograd::Node};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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

    /// Creates a tensor filled with 1.0s.
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        let num_elements = shape.num_elements();
        match dtype {
            DType::F32 => {
                let data = vec![1.0f32; num_elements];
                Self::from_slice(dtype, shape, &data)
            }
            DType::F64 => {
                let data = vec![1.0f64; num_elements];
                Self::from_slice(dtype, shape, &data)
            }
            _ => panic!("Tensor::ones currently only supports F32 and F64"),
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

    /// Forward pass: Sum all elements to a scalar (Autograd aware)
    pub fn sum(&self) -> Tensor {
        let out = crate::kernels::reduce::reduce_forward(self, 0);
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::reduce::ReduceOp {
                orig_shape: self.shape.clone(),
                op_code: 0,
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Forward pass: Mean of all elements to a scalar (Autograd aware)
    pub fn mean(&self) -> Tensor {
        let out = crate::kernels::reduce::reduce_forward(self, 1);
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::reduce::ReduceOp {
                orig_shape: self.shape.clone(),
                op_code: 1,
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
            return self.clone();
        }

        let out = match (&self.device, &device) {
            (Device::Cpu, Device::Gpu(gpu_ctx)) => {
                let data = self.to_contiguous_bytes();
                let alloc = gpu_ctx.arena.allocate(&gpu_ctx.device, data.len() as u64);

                // Use queue.write_buffer instead of manual staging buffers.
                // This eliminates wgpu lifecycle validation errors caused by dropping
                // staging buffers before the GPU finishes processing the command queue.
                gpu_ctx.queue.write_buffer(&alloc.buffer, 0, &data);

                Tensor {
                    id: TensorId::next(),
                    dtype: self.dtype,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    storage: Arc::new(CpuStorage::from_gpu_allocation(alloc)),
                    byte_offset: 0,
                    device: Device::Gpu(gpu_ctx.clone()),
                    requires_grad: self.requires_grad,
                    grad: None,
                    node: None,
                }
            }
            (Device::Gpu(_), Device::Cpu) => {
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
            (Device::Gpu(_), Device::Gpu(_)) => self.clone(),
            _ => panic!("Unsupported device transfer"),
        };

        // Preserve the Autograd graph across PCIe transfers!
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::contiguous::ContiguousOp);
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    fn download_from_gpu(&self) -> Vec<u8> {
        let gpu_ctx = match &self.device {
            Device::Gpu(ctx) => ctx,
            _ => panic!("Not a GPU tensor"),
        };

        let alloc = self.storage.get_gpu_allocation().unwrap();
        let bytes = self.shape.num_elements() * self.dtype.size_in_bytes();

        let staging = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Copy from Cached Buffer -> staging buffer
        encoder.copy_buffer_to_buffer(&alloc.buffer, 0, &staging, 0, bytes as wgpu::BufferAddress); // <--- FIXED offset to 0
        gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

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

    /// Ensures the tensor is on the CPU. If it's on the GPU, it seamlessly downloads it.
    pub fn ensure_cpu(&self) -> Tensor {
        if self.device.is_cpu() {
            self.clone()
        } else {
            self.to(crate::Device::Cpu)
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

    /// Forces a non-contiguous tensor to become contiguous in memory.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }

        let out = if self.device.is_gpu() {
            // GPU STRIDE TRAP:
            // Our GPU kernels assume contiguous memory. If a GPU tensor is non-contiguous
            // (e.g. from a zero-cost transpose), we must download it, physically copy it
            // on the CPU to respect the strides, and upload it back to VRAM.
            let cpu_tensor = self.to(crate::Device::Cpu);
            let cpu_out = Tensor::empty(self.dtype, self.shape.clone());
            crate::kernels::copy(&cpu_tensor, &cpu_out);
            cpu_out.to(self.device.clone())
        } else {
            let cpu_out = Tensor::empty(self.dtype, self.shape.clone());
            crate::kernels::copy(self, &cpu_out);
            cpu_out
        };

        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::contiguous::ContiguousOp);
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
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

        let out = Tensor {
            id: TensorId::next(),
            dtype: self.dtype,
            shape: Shape::new([m, n]),
            strides: Shape::new([m, n]).contiguous_strides(),
            storage: self.storage.clone(),
            byte_offset: self.byte_offset + byte_shift,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            node: None,
        };

        // FIX: Attach the Autograd node if gradients are required!
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::slice::GetSliceOp {
                batch_idx,
                parent_shape: self.shape.clone(),
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
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

    /// 3D Batched Matrix Multiplication with optional transpose and scalar scaling.
    /// Signature: Out[b, i, j] = scale * sum_k(A[b, i, k] * B[b, k, j])
    pub fn batched_matmul(&self, other: &Tensor, transpose_b: bool, scale: f32) -> Tensor {
        let out = if self.rank() == 3
            && other.rank() == 3
            && self.device.is_gpu()
            && other.device.is_gpu()
        {
            let ctx = match &self.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };
            GpuContext::batched_matmul_gpu_resident(&ctx, self, other, transpose_b, scale)
        } else {
            // CPU Fallback (or 2D tensors)
            let cpu_a = self.ensure_cpu();
            let cpu_b = other.ensure_cpu();
            // Simple loop for CPU fallback
            let b_dim = cpu_a.shape.dims()[0];
            let mut out_list = Vec::new();
            for i in 0..b_dim {
                let a_slice = cpu_a.get_2d_slice(i);
                let b_slice = cpu_b.get_2d_slice(i);
                let b_t = if transpose_b { b_slice.t() } else { b_slice };
                let res = crate::kernels::matmul(&a_slice, &b_t);
                out_list.push(crate::kernels::scalar_mul(&res, scale));
            }
            Tensor::cat(&out_list)
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

    pub fn flash_attention(&self, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        let out = if self.device.is_gpu() {
            let ctx = match &self.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };
            GpuContext::flash_attention_gpu(&ctx, self, k, v, scale)
        } else {
            panic!("FlashAttention currently requires GPU");
        };

        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::flash_attention::FlashAttentionOp {
                q: self.clone(),
                k: k.clone(),
                v: v.clone(),
                scale,
            });
            out.with_node(op, vec![self.clone(), k.clone(), v.clone()])
        } else {
            out
        }
    }

    /// Computes the Mean Squared Error between this tensor and a target tensor.
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let loss_val = crate::kernels::mse_loss_forward(self, target);
        let loss_tensor =
            Tensor::from_slice(crate::DType::F32, crate::Shape::new([1]), &[loss_val]);

        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::mse_loss::MSELossOp {
                pred: self.clone(),
                target: target.clone(),
            });
            loss_tensor.with_node(op, vec![self.clone(), target.clone()])
        } else {
            loss_tensor
        }
    }

    pub fn gelu(&self) -> Tensor {
        let out = crate::kernels::gelu(self);
        if self.requires_grad {
            let op = std::sync::Arc::new(crate::ops::gelu::GELUOp {
                input: self.clone(),
            });
            out.with_node(op, vec![self.clone()])
        } else {
            out
        }
    }

    /// Casts the tensor to a different data type.
    /// VETERAN SYSTEMS NOTE: Casting F32 to BF16 truncates the lower 16 bits of the
    /// mantissa. This halves the VRAM footprint and PCIe bandwidth requirements at
    /// the cost of slight precision loss. Casting back to F32 pads the mantissa
    /// with zeros. This method strictly operates on contiguous CPU memory to ensure
    /// byte-level memory layout integrity before GPU upload.
    pub fn to_dtype(&self, target_dtype: crate::DType) -> Tensor {
        use crate::DType;
        use half::bf16;

        if self.dtype == target_dtype {
            return self.clone();
        }

        let cpu_tensor = self.ensure_cpu();

        match (cpu_tensor.dtype, target_dtype) {
            (DType::F32, DType::BF16) => {
                let f32_slice = unsafe { cpu_tensor.storage.as_f32_slice() };
                let mut bf16_bytes = Vec::with_capacity(f32_slice.len() * 2);
                for &val in f32_slice {
                    let b = bf16::from_f32(val);
                    bf16_bytes.extend_from_slice(&b.to_le_bytes());
                }
                let storage =
                    std::sync::Arc::new(crate::storage::CpuStorage::from_bytes(&bf16_bytes));
                Tensor {
                    id: crate::tensor::TensorId::next(),
                    dtype: DType::BF16,
                    shape: cpu_tensor.shape.clone(),
                    strides: cpu_tensor.strides.clone(),
                    storage,
                    byte_offset: 0,
                    device: crate::Device::Cpu,
                    requires_grad: cpu_tensor.requires_grad,
                    grad: None,
                    node: None,
                }
            }
            (DType::BF16, DType::F32) => {
                let bf16_slice = unsafe { cpu_tensor.storage.as_bf16_slice() };
                let mut f32_bytes = Vec::with_capacity(bf16_slice.len() * 4);
                for &val in bf16_slice {
                    let f = val.to_f32();
                    f32_bytes.extend_from_slice(&f.to_le_bytes());
                }
                let storage =
                    std::sync::Arc::new(crate::storage::CpuStorage::from_bytes(&f32_bytes));
                Tensor {
                    id: crate::tensor::TensorId::next(),
                    dtype: DType::F32,
                    shape: cpu_tensor.shape.clone(),
                    strides: cpu_tensor.strides.clone(),
                    storage,
                    byte_offset: 0,
                    device: crate::Device::Cpu,
                    requires_grad: cpu_tensor.requires_grad,
                    grad: None,
                    node: None,
                }
            }
            _ => panic!(
                "Unsupported dtype conversion: {:?} -> {:?}",
                cpu_tensor.dtype, target_dtype
            ),
        }
    }

    /// Computes Cross-Entropy Loss against a batch of target class indices.
    pub fn cross_entropy(&self, targets: &[u32]) -> Tensor {
        let shape = self.shape.dims();
        assert!(
            shape.len() == 2,
            "CrossEntropy expects 2D logits: [Batch, Classes]"
        );

        let batch_size = shape[0];
        let num_classes = shape[1];
        assert_eq!(
            targets.len(),
            batch_size,
            "Targets length must match batch size"
        );

        let logits_cpu = self.ensure_cpu();
        let logits_f32 = unsafe {
            std::slice::from_raw_parts(
                logits_cpu.storage.as_ptr() as *const f32,
                logits_cpu.storage.len() / 4,
            )
        };

        // Run Forward Kernel
        let (loss_val, softmax_probs) = crate::kernels::loss::cross_entropy_forward(
            logits_f32,
            targets,
            batch_size,
            num_classes,
        );

        // Create the Scalar Loss Tensor
        let mut loss_tensor = Tensor::from_slice(DType::F32, Shape::new([1]), &[loss_val]);

        // Attach to Autograd Graph
        if self.requires_grad {
            let op = crate::ops::cross_entropy::CrossEntropyLoss {
                logits: self.clone(),
                targets: targets.to_vec(),
                softmax_probs,
                batch_size,
                num_classes,
            };
            loss_tensor = loss_tensor.with_node(std::sync::Arc::new(op), vec![self.clone()]);
        }

        loss_tensor
    }

    /// Returns a new tensor detached from the current graph.
    /// It shares the same underlying storage but will not track gradients.
    pub fn detach(&self) -> Tensor {
        let mut t = self.clone();
        t.node = None;
        t.requires_grad = false;
        t
    }

    pub fn binary_broadcast(&self, other: &Tensor, op_code: u32) -> Tensor {
        let (out_shape, sa, sb) =
            crate::kernels::broadcast::compute_broadcast(self.shape.dims(), other.shape.dims())
                .expect("Incompatible shapes for broadcasting");

        let out = if self.device.is_gpu() && other.device.is_gpu() {
            let ctx = match &self.device {
                crate::Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };
            crate::gpu::GpuContext::binary_broadcast_gpu(
                &ctx, self, other, op_code, &out_shape, &sa, &sb,
            )
        } else {
            crate::kernels::broadcast::forward_cpu(self, other, op_code, &out_shape, &sa, &sb)
        };

        if self.requires_grad || other.requires_grad {
            let op = std::sync::Arc::new(crate::ops::binary_broadcast::BinaryBroadcastOp {
                a: self.clone(),
                b: other.clone(),
                op_code,
                out_shape,
                strides_a: sa,
                strides_b: sb,
            });
            out.with_node(op, vec![self.clone(), other.clone()])
        } else {
            out
        }
    }

    /// Inherent add method for explicit API calls (e.g., in Linear layers).
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_broadcast(other, 0)
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Tensor {
        self.binary_broadcast(rhs, 0)
    }
}
impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Tensor {
        self.binary_broadcast(rhs, 1)
    }
}
impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor {
        self.binary_broadcast(rhs, 2)
    }
}
impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Tensor {
        self.binary_broadcast(rhs, 3)
    }
}

// Allow mixing owned and referenced tensors
impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        (&self).binary_broadcast(&rhs, 0)
    }
}
impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        (&self).binary_broadcast(&rhs, 1)
    }
}
impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        (&self).binary_broadcast(&rhs, 2)
    }
}
impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        (&self).binary_broadcast(&rhs, 3)
    }
}
