use crate::gpu::arena::GpuAllocation;
use half::bf16;
use memmap2::Mmap;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// VETERAN SYSTEMS ENGINEERING NOTE (Storage Backend Design):
/// The storage backend is the foundation of Axiom's memory model. It must support:
/// 1. CPU-owned memory for kernel execution and data loading
/// 2. Memory-mapped files for zero-copy dataset streaming
/// 3. GPU arena allocations for deterministic VRAM management
/// 4. Mixed-precision dtypes (F32, F64, BF16) with type-safe accessors
///
/// We use an enum-based design with raw byte backing for GPU interop, but provide
/// strongly-typed accessors for CPU-side kernels to ensure mathematical correctness.
#[derive(Debug)]
pub enum Backing {
    /// CPU-owned heap memory. The UnsafeCell allows interior mutability for parallel kernels.
    Owned(UnsafeCell<Vec<u8>>),

    /// Memory-mapped file for zero-copy dataset streaming. Read-only by design.
    Mmap(Arc<Mmap>),

    /// GPU arena allocation. The GpuAllocation struct handles deterministic RAII reclamation.
    Gpu(GpuAllocation),
}

/// The physical storage container for tensor data.
///
/// MEMORY LAYOUT GUARANTEE:
/// All data is stored in little-endian byte order, tightly packed with no padding.
/// - F32: 4 bytes per element, IEEE 754 binary32 format
/// - F64: 8 bytes per element, IEEE 754 binary64 format  
/// - BF16: 2 bytes per element, BFloat16 format (8-bit exponent, 7-bit mantissa)
///
/// SAFETY INVARIANT:
/// The `data` field is wrapped in UnsafeCell to enable interior mutability for parallel
/// CPU kernels (Rayon). Concurrent access is managed manually by the kernel implementations
/// via SyncPtr/SyncMutPtr wrappers. The caller is responsible for ensuring no data races.
#[derive(Debug)]
pub struct CpuStorage {
    data: Backing,
}

// SAFETY: CpuStorage is Send and Sync because:
// 1. Backing::Owned uses UnsafeCell<Vec<u8>> which is !Sync, but we manually manage
//    concurrent access via SyncPtr wrappers in the kernel layer.
// 2. Backing::Mmap uses Arc<Mmap> which is thread-safe for reads.
// 3. Backing::Gpu uses Arc<GpuAllocation> which is thread-safe.
// The caller must ensure no simultaneous mutable access to the same memory region.
unsafe impl Send for CpuStorage {}
unsafe impl Sync for CpuStorage {}

impl CpuStorage {
    /// Allocates zero-initialized CPU memory of the specified byte size.
    ///
    /// VETERAN NOTE: This is the primary allocation path for CPU-side tensors.
    /// The Vec<u8> is heap-allocated and will be automatically deallocated by Rust's
    /// drop glue when the CpuStorage is destroyed, unless it has been moved into
    /// a GpuAllocation via the Arena.
    pub fn zeros(bytes: usize) -> Self {
        Self {
            data: Backing::Owned(UnsafeCell::new(vec![0; bytes])),
        }
    }

    /// Allocates uninitialized CPU memory of the specified byte size.
    ///
    /// SAFETY: The returned memory contains arbitrary garbage values. The caller
    /// must fully initialize the memory before reading from it. This is used for
    /// performance-critical paths where zeroing is unnecessary (e.g., immediately
    /// overwriting with kernel output).
    pub fn empty(bytes: usize) -> Self {
        let mut data = Vec::with_capacity(bytes);
        unsafe {
            data.set_len(bytes);
        }
        Self {
            data: Backing::Owned(UnsafeCell::new(data)),
        }
    }

    /// Creates storage from an existing byte slice (deep copy).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: Backing::Owned(UnsafeCell::new(bytes.to_vec())),
        }
    }

    /// Creates storage from a memory-mapped file (zero-copy, read-only).
    ///
    /// VETERAN SYSTEMS NOTE: Memory-mapped datasets enable training on datasets
    /// larger than system RAM by leveraging the OS page cache. The Mmap is wrapped
    /// in Arc to allow safe sharing across multiple tensor views without copying.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
        Self {
            data: Backing::Mmap(mmap),
        }
    }

    /// Creates storage backed by a GPU arena allocation.
    ///
    /// MEMORY LIFECYCLE GUARANTEE:
    /// The GpuAllocation struct contains a Weak reference to the GpuMemoryArena.
    /// When this CpuStorage is dropped, if the GpuAllocation is the last reference
    /// to the underlying GPU buffer (Arc::strong_count == 1), the buffer is returned
    /// to the arena's free list for immediate reuse. This ensures deterministic VRAM
    /// reclamation with zero garbage collection pauses.
    pub fn from_gpu_allocation(alloc: GpuAllocation) -> Self {
        Self {
            data: Backing::Gpu(alloc),
        }
    }

    /// Returns a raw const pointer to the underlying byte data.
    ///
    /// SAFETY: The returned pointer is valid for the lifetime of the CpuStorage.
    /// The caller must ensure:
    /// 1. No mutable access occurs while this pointer is in use
    /// 2. The pointer is not used after the CpuStorage is dropped
    /// 3. For GPU-backed storage, this method panics - use .to(Cpu) first
    pub fn as_ptr(&self) -> *const u8 {
        match &self.data {
            Backing::Owned(cell) => unsafe { (*cell.get()).as_ptr() },
            Backing::Mmap(m) => m.as_ptr(),
            Backing::Gpu(_) => {
                panic!("Cannot get CPU pointer from GPU buffer! Call .to(Cpu) first.")
            }
        }
    }

    /// Returns a raw mutable pointer to the underlying byte data.
    ///
    /// SAFETY: The returned pointer grants exclusive mutable access. The caller
    /// must ensure no other references (const or mutable) exist to this memory
    /// region while the pointer is in use. This is the foundation of our parallel
    /// CPU kernels - they use SyncMutPtr to safely share mutable access across
    /// Rayon threads.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        match &self.data {
            Backing::Owned(cell) => unsafe { (*cell.get()).as_mut_ptr() },
            Backing::Mmap(_) => panic!("Cannot mutate a read-only memory-mapped dataset!"),
            Backing::Gpu(_) => panic!("Cannot get mutable CPU pointer from GPU buffer!"),
        }
    }

    /// Returns true if this storage is backed by GPU VRAM.
    pub fn is_gpu(&self) -> bool {
        matches!(&self.data, Backing::Gpu(_))
    }

    /// Extracts a reference to the GPU allocation if this storage is GPU-backed.
    pub fn get_gpu_allocation(&self) -> Option<&GpuAllocation> {
        match &self.data {
            Backing::Gpu(alloc) => Some(alloc),
            _ => None,
        }
    }

    // ========================================================================
    // TYPE-SAFE ACCESSORS FOR MIXED-PRECISION SUPPORT (RUST 2024 COMPLIANT)
    // ========================================================================

    /// Returns a typed slice for F32 data.
    ///
    /// SAFETY: The caller must guarantee that:
    /// 1. The storage is CPU-owned (not GPU-backed)
    /// 2. The total byte size is divisible by 4 (size of f32)
    /// 3. The data is properly aligned for f32 access
    /// 4. No mutable access occurs while this slice is in use
    pub unsafe fn as_f32_slice(&self) -> &[f32] {
        match &self.data {
            Backing::Owned(cell) => {
                // VETERAN SYSTEMS NOTE (Rust 2024 Compliance):
                // Even though this function is marked `unsafe fn`, Rust 2024 requires
                // explicit `unsafe {}` blocks for the actual unsafe operations inside.
                // This strictly isolates the pointer dereference and slice creation,
                // preventing accidental Undefined Behavior if safe logic is mixed in.
                unsafe {
                    let bytes = &*cell.get();
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const f32,
                        bytes.len() / std::mem::size_of::<f32>(),
                    )
                }
            }
            Backing::Mmap(m) => unsafe {
                std::slice::from_raw_parts(
                    m.as_ptr() as *const f32,
                    m.len() / std::mem::size_of::<f32>(),
                )
            },
            Backing::Gpu(_) => panic!("Cannot access GPU storage as f32 slice!"),
        }
    }

    /// Returns a typed mutable slice for F32 data.
    ///
    /// SAFETY: The caller must guarantee exclusive mutable access to the memory
    /// region. This is used by parallel CPU kernels with SyncMutPtr to safely
    /// distribute mutable access across threads.
    pub unsafe fn as_f32_slice_mut(&self) -> &mut [f32] {
        match &self.data {
            Backing::Owned(cell) => unsafe {
                let bytes = &mut *cell.get();
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            },
            Backing::Mmap(_) => panic!("Cannot mutate memory-mapped storage!"),
            Backing::Gpu(_) => panic!("Cannot access GPU storage as mutable f32 slice!"),
        }
    }

    /// Returns a typed slice for BF16 data.
    ///
    /// VETERAN NOTE: BFloat16 uses 2 bytes per element with the same exponent
    /// range as F32. This accessor enables type-safe CPU-side operations on
    /// BF16 tensors (e.g., casting, debugging, loss computation) while maintaining
    /// the memory layout expected by GPU shaders.
    pub unsafe fn as_bf16_slice(&self) -> &[bf16] {
        match &self.data {
            Backing::Owned(cell) => unsafe {
                let bytes = &*cell.get();
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const bf16,
                    bytes.len() / std::mem::size_of::<bf16>(),
                )
            },
            Backing::Mmap(m) => unsafe {
                std::slice::from_raw_parts(
                    m.as_ptr() as *const bf16,
                    m.len() / std::mem::size_of::<bf16>(),
                )
            },
            Backing::Gpu(_) => panic!("Cannot access GPU storage as bf16 slice!"),
        }
    }

    /// Returns a typed mutable slice for BF16 data.
    pub unsafe fn as_bf16_slice_mut(&self) -> &mut [bf16] {
        match &self.data {
            Backing::Owned(cell) => unsafe {
                let bytes = &mut *cell.get();
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut bf16,
                    bytes.len() / std::mem::size_of::<bf16>(),
                )
            },
            Backing::Mmap(_) => panic!("Cannot mutate memory-mapped storage!"),
            Backing::Gpu(_) => panic!("Cannot access GPU storage as mutable bf16 slice!"),
        }
    }

    /// Returns the total byte size of the storage.
    pub fn len(&self) -> usize {
        match &self.data {
            Backing::Owned(cell) => unsafe { (*cell.get()).len() },
            Backing::Mmap(m) => m.len(),
            Backing::Gpu(alloc) => alloc.size as usize,
        }
    }

    /// Returns true if the storage contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
