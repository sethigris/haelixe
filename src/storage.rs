use crate::gpu::arena::GpuAllocation;
use memmap2::Mmap;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// The physical backing of the memory.
#[derive(Debug)]
pub enum Backing {
    Owned(UnsafeCell<Vec<u8>>),
    Mmap(Arc<Mmap>),
    Gpu(GpuAllocation), // Uses the new Arena Allocation
}

#[derive(Debug)]
pub struct CpuStorage {
    data: Backing,
}

// SAFETY: We manage concurrent access manually.
unsafe impl Send for CpuStorage {}
unsafe impl Sync for CpuStorage {}

impl CpuStorage {
    pub fn zeros(bytes: usize) -> Self {
        Self {
            data: Backing::Owned(UnsafeCell::new(vec![0; bytes])),
        }
    }

    pub fn empty(bytes: usize) -> Self {
        let mut data = Vec::with_capacity(bytes);
        unsafe {
            data.set_len(bytes);
        }
        Self {
            data: Backing::Owned(UnsafeCell::new(data)),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: Backing::Owned(UnsafeCell::new(bytes.to_vec())),
        }
    }

    pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
        Self {
            data: Backing::Mmap(mmap),
        }
    }

    /// Creates a CpuStorage backed by a slice of the GPU Memory Arena
    pub fn from_gpu_allocation(alloc: GpuAllocation) -> Self {
        Self {
            data: Backing::Gpu(alloc),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        match &self.data {
            Backing::Owned(cell) => unsafe { (*cell.get()).as_ptr() },
            Backing::Mmap(m) => m.as_ptr(),
            Backing::Gpu(_) => {
                panic!("Cannot get CPU pointer from GPU buffer! Call .to(Cpu) first.")
            }
        }
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        match &self.data {
            Backing::Owned(cell) => unsafe { (*cell.get()).as_mut_ptr() },
            Backing::Mmap(_) => panic!("Cannot mutate a read-only memory-mapped dataset!"),
            Backing::Gpu(_) => panic!("Cannot get mutable CPU pointer from GPU buffer!"),
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(&self.data, Backing::Gpu(_))
    }

    /// Extracts the GPU Allocation from the storage backend
    pub fn get_gpu_allocation(&self) -> Option<&GpuAllocation> {
        match &self.data {
            Backing::Gpu(alloc) => Some(alloc),
            _ => None,
        }
    }
}
