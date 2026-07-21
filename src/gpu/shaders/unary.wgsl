// --------------------------------------------------------------------------
// Module: gpu::shaders::unary
// --------------------------------------------------------------------------
//
// PURPOSE:
//   Fused WGSL compute shader for unary element-wise operations.
//   Routes math operations via an `op_code` uniform to maximize GPU
//   pipeline residency and minimize binary bloat.
//
// HISTORICAL CONTEXT:
//   Forged during Phase 1 of the 1.2M LOC Kernel Zoo expansion.
//   Replaces the need for 7 separate compute pipelines, keeping the
//   WebGPU command encoder warm and reducing dispatch overhead.
//
// MATHEMATICAL INVARIANTS:
//   - Op Codes: 0=Exp, 1=Log, 2=Sqrt, 3=Tanh, 4=ReLU, 5=GELU, 6=SiLU
//   - GELU uses the Tanh approximation for cross-platform stability.
//
// AUTHORSHIP:
//   Engineered by Sethigris and the Haelixe core team.
//   Date: 2026-07-22
// --------------------------------------------------------------------------

struct Params { total: u32, op_code: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) { return; }

    let x = input[idx];
    var res: f32;

    switch (params.op_code) {
        case 0u: { res = exp(x); }
        case 1u: { res = log(x); }
        case 2u: { res = sqrt(x); }
        case 3u: { res = tanh(x); }
        case 4u: { res = max(x, 0.0); } // ReLU
        case 5u: { // GELU (Tanh Approximation)
            let c = 0.7978845608; // sqrt(2.0 / PI)
            let inner = c * (x + 0.044715 * x * x * x);
            res = 0.5 * x * (1.0 + tanh(inner));
        }
        case 6u: { // SiLU (Swish)
            let sig = 1.0 / (1.0 + exp(-x));
            res = x * sig;
        }
        default: { res = 0.0; }
    }

    output[idx] = res;
}
