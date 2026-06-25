use crate::device::Device;
use crate::{CpuStorage, DType, Shape, Tensor, TensorId};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub mod arena;

#[derive(Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bind_group_layout: wgpu::BindGroupLayout,
    fused_linear_pipeline: wgpu::ComputePipeline,
    fused_linear_bind_group_layout: wgpu::BindGroupLayout,
    sgd_pipeline: wgpu::ComputePipeline,
    sgd_bind_group_layout: wgpu::BindGroupLayout,
    pub arena: arena::GpuMemoryArena,
}

impl GpuContext {
    /// Initializes the GPU. This is slow (seconds) on first call because it
    /// negotiates with the driver and compiles shaders. Cache this instance!
    pub fn new() -> Arc<Self> {
        // 1. Create a wgpu instance (detects available backends: Vulkan/Metal/DX12)
        let instance = wgpu::Instance::default();

        // 2. Request an adapter (physical GPU). pollster bridges async -> sync.
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("No GPU adapter found! Is a GPU available?");

        println!("Using GPU adapter: {:?}", adapter.get_info().name);

        // 3. Request a logical device and command queue
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("Failed to create GPU device");

        // 4. Compile the WGSL shader into a GPU-executable module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
        });

        // 5. Define the bind group layout (what buffers the shader expects)
        let add_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("add_bind_group_layout"),
                entries: &[
                    // Binding 0: tensor A (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: tensor B (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: output tensor (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let arena = arena::GpuMemoryArena::new(&device, 256); // 256MB Arena 

        // 6. Create the pipeline layout and the compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("add_pipeline_layout"),
            bind_group_layouts: &[&add_bind_group_layout],
            push_constant_ranges: &[],
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // --- MatMul Pipeline Setup ---
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let matmul_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul_bind_group_layout"),
                entries: &[
                    // Binding 0: matrix A (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: matrix B (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: output matrix (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: params uniform buffer (M, K, N dimensions)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let matmul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&matmul_bind_group_layout],
                push_constant_ranges: &[],
            });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // --- Fused Linear Pipeline (MatMul + Bias + ReLU) ---
        let fused_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_linear_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/linear_fused.wgsl").into()),
        });

        let fused_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fused_linear_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let fused_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fused_linear_layout"),
                bind_group_layouts: &[&fused_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fused_linear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fused_linear_pipeline"),
                layout: Some(&fused_pipeline_layout),
                module: &fused_shader,
                entry_point: "main",
                compilation_options: Default::default(),
            });

        // --- SGD Pipeline ---
        let sgd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sgd.wgsl").into()),
        });

        let sgd_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sgd_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sgd_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sgd_layout"),
            bind_group_layouts: &[&sgd_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sgd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgd_pipeline"),
            layout: Some(&sgd_pipeline_layout),
            module: &sgd_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Arc::new(Self {
            device,
            queue,
            add_pipeline,
            add_bind_group_layout,
            matmul_pipeline,
            matmul_bind_group_layout,
            fused_linear_pipeline,
            fused_linear_bind_group_layout: fused_bind_group_layout,
            sgd_pipeline,
            sgd_bind_group_layout: sgd_bind_group_layout,
            arena,
        })
    }

    /// Element-wise addition on the GPU.
    /// Uploads tensors A and B to VRAM, runs the shader, downloads the result.
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape, b.shape, "GPU add requires matching shapes");
        assert_eq!(a.dtype, DType::F32, "GPU add currently only supports F32");

        let num_elements = a.shape.num_elements();
        let bytes = num_elements * std::mem::size_of::<f32>();

        // Materialize the CPU tensor data into contiguous byte slices.
        // (In a real framework, we'd keep data on GPU and avoid this transfer.)
        let a_data = tensor_to_bytes_f32(a);
        let b_data = tensor_to_bytes_f32(b);

        // 7. Create GPU buffers in VRAM and upload the data
        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a_buffer"),
                contents: &a_data,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b_buffer"),
                contents: &b_data,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer: must be STORAGE | COPY_SRC so we can read it back
        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buffer"),
            size: bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 8. Create a bind group: binds our 3 buffers to the shader's 3 bindings
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bind_group"),
            layout: &self.add_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        // 9. Record commands: begin compute pass, set pipeline, dispatch, end
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("add_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.add_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups. Each workgroup has 256 threads.
            // We need ceil(num_elements / 256) workgroups to cover all elements.
            let workgroup_count = ((num_elements + 255) / 256) as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // 10. Create a staging buffer to read results back to CPU.
        // It needs MAP_READ | COPY_DST usage.
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU output buffer -> CPU staging buffer
        encoder.copy_buffer_to_buffer(
            &out_buffer,
            0,
            &staging_buffer,
            0,
            bytes as wgpu::BufferAddress,
        );

        // 11. Submit the commands to the GPU queue
        self.queue.submit(std::iter::once(encoder.finish()));

        // 12. Map the staging buffer into CPU address space and read the data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result_bytes = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        // 13. Wrap the result bytes back into a CPU Tensor
        Tensor {
            id: TensorId::next(),
            dtype: DType::F32,
            shape: a.shape.clone(),
            strides: a.shape.contiguous_strides(),
            storage: Arc::new(CpuStorage::from_bytes(&result_bytes)),
            byte_offset: 0,
            requires_grad: false,
            grad: None,
            node: None,
            device: crate::device::Device::Cpu,
        }
    }

    pub fn add_gpu_resident(ctx: &Arc<GpuContext>, a: &Tensor, b: &Tensor) -> Tensor {
        assert!(
            a.storage.is_gpu() && b.storage.is_gpu(),
            "Both tensors must be on GPU"
        );
        assert_eq!(a.shape, b.shape);

        let bytes = a.shape.num_elements() * std::mem::size_of::<f32>();

        let a_alloc = a.storage.get_gpu_allocation().unwrap();
        let b_alloc = b.storage.get_gpu_allocation().unwrap();

        // ARENA ALLOCATION
        let out_alloc = ctx.arena.allocate(bytes as u64);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.add_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_alloc.as_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.add_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((a.shape.num_elements() + 255) / 256) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        Tensor {
            id: TensorId::next(),
            dtype: DType::F32,
            shape: a.shape.clone(),
            strides: a.shape.contiguous_strides(),
            storage: Arc::new(CpuStorage::from_gpu_allocation(out_alloc)),
            byte_offset: 0,
            device: Device::Gpu(ctx.clone()),
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    /// GPU-resident matrix multiplication: C = A @ B
    /// All inputs and outputs stay on GPU. Uses tiled shared memory for performance.

    pub fn matmul_gpu_resident(ctx: &Arc<GpuContext>, a: &Tensor, b: &Tensor) -> Tensor {
        assert!(
            a.storage.is_gpu() && b.storage.is_gpu(),
            "Both tensors must be on GPU"
        );
        assert_eq!(a.rank(), 2);
        assert_eq!(b.rank(), 2);

        let m = a.shape.dims()[0];
        let k = a.shape.dims()[1];
        let k_b = b.shape.dims()[0];
        let n = b.shape.dims()[1];
        assert_eq!(k, k_b, "Inner dimensions must match");

        let out_bytes = m * n * std::mem::size_of::<f32>();

        let a_alloc = a.storage.get_gpu_allocation().unwrap();
        let b_alloc = b.storage.get_gpu_allocation().unwrap();

        // ARENA ALLOCATION
        let out_alloc = ctx.arena.allocate(out_bytes as u64);

        let params_data: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &ctx.matmul_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = ((n + 15) / 16) as u32;
            let wg_y = ((m + 15) / 16) as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        Tensor {
            id: TensorId::next(),
            dtype: DType::F32,
            shape: Shape::new([m, n]),
            strides: Shape::new([m, n]).contiguous_strides(),
            storage: Arc::new(CpuStorage::from_gpu_allocation(out_alloc)),
            byte_offset: 0,
            device: Device::Gpu(ctx.clone()),
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    pub fn fused_linear(ctx: &Arc<GpuContext>, x: &Tensor, w: &Tensor, bias: &Tensor) -> Tensor {
        assert!(
            x.storage.is_gpu() && w.storage.is_gpu() && bias.storage.is_gpu(),
            "All inputs must be on GPU"
        );

        let m = x.shape.dims()[0];
        let k = x.shape.dims()[1];
        let n = w.shape.dims()[1];
        let out_bytes = m * n * std::mem::size_of::<f32>();

        let x_alloc = x.storage.get_gpu_allocation().unwrap();
        let w_alloc = w.storage.get_gpu_allocation().unwrap();
        let bias_alloc = bias.storage.get_gpu_allocation().unwrap();

        // ARENA ALLOCATION
        let out_alloc = ctx.arena.allocate(out_bytes as u64);

        let params_data: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_bg"),
            layout: &ctx.fused_linear_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.fused_linear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        Tensor {
            id: TensorId::next(),
            dtype: DType::F32,
            shape: Shape::new([m, n]),
            strides: Shape::new([m, n]).contiguous_strides(),
            storage: Arc::new(CpuStorage::from_gpu_allocation(out_alloc)),
            byte_offset: 0,
            device: Device::Gpu(ctx.clone()),
            requires_grad: false,
            grad: None,
            node: None,
        }
    }

    pub fn sgd_step_gpu(ctx: &Arc<GpuContext>, weight: &Tensor, grad: &Tensor, lr: f32) {
        assert!(
            weight.storage.is_gpu() && grad.storage.is_gpu(),
            "Both weight and grad must be on GPU for GPU SGD"
        );
        assert_eq!(weight.shape, grad.shape);

        let w_alloc = weight.storage.get_gpu_allocation().unwrap();
        let g_alloc = grad.storage.get_gpu_allocation().unwrap();

        let params_data: [f32; 4] = [lr, 0.0, 0.0, 0.0];
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sgd_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sgd_bg"),
            layout: &ctx.sgd_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: g_alloc.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.sgd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((weight.shape.num_elements() + 255) / 256) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Helper: Materialize a tensor's data into a contiguous byte Vec,
/// respecting arbitrary strides (handles sliced/transposed tensors).
fn tensor_to_bytes_f32(tensor: &Tensor) -> Vec<u8> {
    let num_elements = tensor.shape.num_elements();
    let mut result = Vec::with_capacity(num_elements * 4);

    let shape = tensor.shape.dims();
    let strides = tensor.strides.steps();
    let base = tensor.byte_offset / std::mem::size_of::<f32>();
    let ptr = tensor.storage.as_ptr() as *const f32;

    // Flatten N-D space into 1-D, respecting strides
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
