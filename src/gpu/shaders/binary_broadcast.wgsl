// --------------------------------------------------------------------------
// Module: gpu::shaders::binary_broadcast
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Fused WGSL compute shader for element-wise binary operations with
//   N-dimensional broadcasting. Uses "virtual strides" to read from
//   input buffers without physical memory expansion.
//
// MATHEMATICAL INVARIANTS:
//   - Supports up to 6 dimensions (sufficient for 99% of LLM tensors).
//   - Op Codes: 0=Add, 1=Mul, 2=Sub, 3=Div
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-20
// --------------------------------------------------------------------------

struct Meta {
    // Output shape and strides
    out_shape: vec4<u32>, // dims 0..3
    out_shape_2: vec2<u32>, // dims 4..5
    out_strides: vec4<u32>,
    out_strides_2: vec2<u32>,
    
    // Input A virtual strides
    a_strides: vec4<u32>,
    a_strides_2: vec2<u32>,
    
    // Input B virtual strides
    b_strides: vec4<u32>,
    b_strides_2: vec2<u32>,
    
    total_elements: u32,
    op_code: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: Meta;

// Helper to calculate the flat memory offset for a given N-dim index
fn get_offset(idx: vec4<u32>, idx_2: vec2<u32>, strides: vec4<u32>, strides_2: vec2<u32>) -> u32 {
    return idx.x * strides.x + idx.y * strides.y + idx.z * strides.z + idx.w * strides.w 
         + idx_2.x * strides_2.x + idx_2.y * strides_2.y;
}

// Helper to convert flat 1D index to N-dim coordinates
fn flat_to_nd(flat_idx: u32) -> (vec4<u32>, vec2<u32>) {
    var rem = flat_idx;
    // Dims are processed from last to first (row-major)
    let d5 = rem % meta.out_shape_2.y; rem = rem / meta.out_shape_2.y;
    let d4 = rem % meta.out_shape_2.x; rem = rem / meta.out_shape_2.x;
    let d3 = rem % meta.out_shape.w;   rem = rem / meta.out_shape.w;
    let d2 = rem % meta.out_shape.z;   rem = rem / meta.out_shape.z;
    let d1 = rem % meta.out_shape.y;   rem = rem / meta.out_shape.y;
    let d0 = rem; // meta.out_shape.x
    
    return (vec4<u32>(d0, d1, d2, d3), vec2<u32>(d4, d5));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= meta.total_elements) { return; }

    let (nd_idx, nd_idx_2) = flat_to_nd(idx);

    // Calculate memory offsets using virtual strides
    let off_a = get_offset(nd_idx, nd_idx_2, meta.a_strides, meta.a_strides_2);
    let off_b = get_offset(nd_idx, nd_idx_2, meta.b_strides, meta.b_strides_2);

    let val_a = a[off_a];
    let val_b = b[off_b];

    var res: f32;
    switch (meta.op_code) {
        case 0u: { res = val_a + val_b; }
        case 1u: { res = val_a * val_b; }
        case 2u: { res = val_a - val_b; }
        case 3u: { res = val_a / val_b; }
        default: { res = 0.0; }
    }

    out[idx] = res;
}