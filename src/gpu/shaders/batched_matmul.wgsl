const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// Uniforms: batch, m, k, n, scale, transpose_b, pad, pad
struct Params { 
    batch: u32, m: u32, k: u32, n: u32, 
    scale: f32, transpose_b: u32, _p1: u32, _p2: u32 
};
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(TILE_M, TILE_N, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_idx = wg_id.z;
    let row = wg_id.y * TILE_M + local_id.y;
    let col = wg_id.x * TILE_N + local_id.x;

    let a_offset = batch_idx * params.m * params.k;
    let b_offset = batch_idx * params.k * params.n; // Note: if transposed, memory is actually [B, N, K]
    let out_offset = batch_idx * params.m * params.n;

    var sum: f32 = 0.0;
    let num_k_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        let a_col = t * TILE_K + local_id.x;
        if (row < params.m && a_col < params.k) {
            tile_a[local_id.y * TILE_K + local_id.x] = a[a_offset + row * params.k + a_col];
        } else {
            tile_a[local_id.y * TILE_K + local_id.x] = 0.0;
        }

        let b_row = t * TILE_K + local_id.y; // This is the K index
        let b_col = col;                     // This is the N index
        
        var b_val: f32 = 0.0;
        if (b_row < params.k && b_col < params.n) {
            if (params.transpose_b == 1u) {
                // B is stored as [B, N, K]. We want element at [batch, k_idx, n_idx]
                b_val = b[b_offset + b_col * params.k + b_row];
            } else {
                // B is stored as [B, K, N].
                b_val = b[b_offset + b_row * params.n + b_col];
            }
            tile_b[local_id.y * TILE_N + local_id.x] = b_val;
        } else {
            tile_b[local_id.y * TILE_N + local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_K; i = i + 1u) {
            sum = sum + tile_a[local_id.y * TILE_K + i] * tile_b[i * TILE_N + local_id.x];
        }
        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        out[out_offset + row * params.n + col] = sum * params.scale;
    }
}