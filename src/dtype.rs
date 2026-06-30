/// Hardware-supported data types.
/// Kept as a simple enum to allow fast matching and easy FFI translation later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    /// Returns the size of a single element in bytes.
    /// Used heavily for raw pointer arithmetic and memory allocation.
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            DType::U8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
}
