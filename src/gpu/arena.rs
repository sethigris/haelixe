use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use wgpu::{BindingResource, Buffer, BufferBinding, BufferUsages, Device};

pub struct GpuMemoryArena {
    free_list: Mutex<HashMap<u64, Vec<Arc<Buffer>>>>,
}

impl GpuMemoryArena {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            free_list: Mutex::new(HashMap::new()),
        })
    }

    pub fn allocate(self: &Arc<Self>, device: &Device, requested_bytes: u64) -> GpuAllocation {
        let binned_size = get_bin_size(requested_bytes);

        let mut cache = self.free_list.lock().unwrap();
        if let Some(buffers) = cache.get_mut(&binned_size) {
            if let Some(buffer) = buffers.pop() {
                return GpuAllocation {
                    buffer,
                    size: binned_size,
                    arena: Arc::downgrade(self),
                };
            }
        }
        drop(cache);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuMemoryArena Binned Slab"),
            size: binned_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuAllocation {
            buffer: Arc::new(buffer),
            size: binned_size,
            arena: Arc::downgrade(self),
        }
    }

    pub fn free(&self, buffer: Arc<Buffer>, binned_size: u64) {
        let mut cache = self.free_list.lock().unwrap();
        cache
            .entry(binned_size)
            .or_insert_with(Vec::new)
            .push(buffer);
    }

    pub fn reset(&self) {
        self.free_list.lock().unwrap().clear();
    }
}

fn get_bin_size(requested_bytes: u64) -> u64 {
    const MIN_BIN_SIZE: u64 = 512;
    if requested_bytes <= MIN_BIN_SIZE {
        MIN_BIN_SIZE
    } else {
        requested_bytes.next_power_of_two()
    }
}

impl std::fmt::Debug for GpuMemoryArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.free_list.lock().unwrap();
        let total_cached_buffers: usize = cache.values().map(|v| v.len()).sum();
        f.debug_struct("GpuMemoryArena")
            .field("cached_unique_bins", &cache.keys().len())
            .field("total_free_slabs", &total_cached_buffers)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub buffer: Arc<Buffer>,
    pub size: u64,
    arena: Weak<GpuMemoryArena>,
}

impl Drop for GpuAllocation {
    fn drop(&mut self) {
        if Arc::strong_count(&self.buffer) == 1 {
            if let Some(arena) = self.arena.upgrade() {
                arena.free(self.buffer.clone(), self.size);
            }
        }
    }
}

impl GpuAllocation {
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}
