use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::{BindingResource, Buffer, BufferBinding, BufferUsages, Device};

pub struct GpuMemoryArena {
    master_buffer: Arc<Buffer>,
    capacity: u64,
    offset: AtomicU64,
}

impl GpuMemoryArena {
    pub fn new(device: &Device, capacity_mb: u64) -> Self {
        let capacity_bytes = capacity_mb * 1024 * 1024;
        let master_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuMemoryArena Master Block"),
            size: capacity_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            master_buffer: Arc::new(master_buffer),
            capacity: capacity_bytes,
            offset: AtomicU64::new(0),
        }
    }

    /// O(1) VRAM allocation.
    pub fn allocate(&self, size_bytes: u64) -> GpuAllocation {
        // wgpu requires storage buffer bindings to be aligned to 256 bytes
        let alignment = 256;
        let aligned_size = (size_bytes + alignment - 1) / alignment * alignment;

        let current_offset = self.offset.fetch_add(aligned_size, Ordering::SeqCst);

        if current_offset + aligned_size > self.capacity {
            panic!(
                "GpuMemoryArena Out Of Memory! Requested {} bytes.",
                aligned_size
            );
        }

        GpuAllocation {
            buffer: self.master_buffer.clone(),
            offset: current_offset,
            size: size_bytes,
        }
    }

    /// Resets the arena. Call this at the end of a training step to reclaim VRAM!
    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub buffer: Arc<Buffer>,
    pub offset: u64,
    pub size: u64,
}

impl GpuAllocation {
    /// Helper to easily create a wgpu BindingResource for compute shaders
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

impl std::fmt::Debug for GpuMemoryArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryArena")
            .field("capacity_mb", &(self.capacity / (1024 * 1024)))
            .field("current_offset_bytes", &self.offset.load(Ordering::SeqCst))
            .finish()
    }
}
