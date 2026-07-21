// --------------------------------------------------------------------------
// Module: gpu::shaders::reduce
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Fused WGSL compute shader for global tensor reduction (Sum/Mean).
//   Uses a logarithmic tree-reduction in L1 Workgroup Memory (SRAM) to
//   bypass the lack of native `atomic<f32>` in core WebGPU.
//
// HISTORICAL CONTEXT:
//   Forged to close the gap between toy frameworks and production engines.
//   Without reductions, Loss functions and Normalization layers cannot
//   be executed natively on the silicon.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-21
// --------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> wg_sums: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x=total_elements

var<workgroup> s_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    let tid = lid.x;
    let total = params.x;

    // 1. Load into SRAM (Guard against out-of-bounds)
    if (idx < total) {
        s_data[tid] = input[idx];
    } else {
        s_data[tid] = 0.0;
    }
    workgroupBarrier();

    // 2. Logarithmic Tree-Reduction in SRAM
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride && (idx + stride) < total) {
            s_data[tid] = s_data[tid] + s_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // 3. Write the single workgroup sum to the intermediate buffer
    if (tid == 0u) {
        wg_sums[wid.x] = s_data[0];
    }
}
