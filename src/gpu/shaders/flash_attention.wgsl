const Br: u32 = 16u; // Query block size
const Bc: u32 = 16u; // Key/Value block size
const D: u32 = 64u;  // Head dimension

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { batch_heads: u32, seq_len: u32, head_dim: u32, scale: f32 };
@group(0) @binding(4) var<uniform> params: Params;

// L1 Cache (Workgroup Shared Memory)
var<workgroup> s_q: array<f32, 1024>; // 16 * 64
var<workgroup> s_k: array<f32, 1024>; // 16 * 64
var<workgroup> s_v: array<f32, 1024>; // 16 * 64
var<workgroup> s_p: array<f32, 256>;  // 16 * 16

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let bh = wg_id.x; 
    let q_block_idx = wg_id.y; 
    let tid = local_id.x; // 0 to 255
    let q_start = q_block_idx * Br;
    
    // Registers for Online Softmax (Each thread tracks 4 output elements)
    var m_i: array<f32, 4>; 
    var l_i: array<f32, 4>; 
    var o_i: array<f32, 4>; 
    
    for (var i = 0u; i < 4u; i = i + 1u) {
        m_i[i] = -3.402823e+38f; // -inf
        l_i[i] = 0.0;
        o_i[i] = 0.0;
    }
    
    // 1. Load Q block into L1 Cache
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid * 4u + i;
        let r = idx / D;
        let c = idx % D;
        let global_q_idx = (bh * params.seq_len + q_start + r) * D + c;
        if (q_start + r < params.seq_len) { s_q[idx] = q[global_q_idx]; } 
        else { s_q[idx] = 0.0; }
    }
    workgroupBarrier();
    
    let num_k_blocks = (params.seq_len + Bc - 1u) / Bc;
    
    // 2. Loop over K/V blocks
    for (var kb = 0u; kb < num_k_blocks; kb = kb + 1u) {
        let k_start = kb * Bc;
        
        // Load K and V blocks into L1 Cache
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid * 4u + i;
            let r = idx / D;
            let c = idx % D;
            let global_kv_idx = (bh * params.seq_len + k_start + r) * D + c;
            if (k_start + r < params.seq_len) {
                s_k[idx] = k[global_kv_idx];
                s_v[idx] = v[global_kv_idx];
            } else {
                s_k[idx] = 0.0;
                s_v[idx] = 0.0;
            }
        }
        workgroupBarrier();
        
        // Compute S = Q * K^T (16x16 matrix) in L1 Cache
        if (tid < 256u) {
            let row = tid / Bc; 
            let col = tid % Bc; 
            var sum = 0.0;
            for (var d = 0u; d < D; d = d + 1u) {
                sum = sum + s_q[row * D + d] * s_k[col * D + d];
            }
            s_p[tid] = sum * params.scale;
        }
        workgroupBarrier();
        
        // Online Softmax & Output Accumulation
        for (var i = 0u; i < 4u; i = i + 1u) {
            let out_idx = tid * 4u + i;
            let row = out_idx / D; 
            let col = out_idx % D;
            
            var row_max = m_i[i];
            for (var j = 0u; j < Bc; j = j + 1u) {
                row_max = max(row_max, s_p[row * Bc + j]);
            }
            
            let rescale = exp(m_i[i] - row_max);
            var p_sum = 0.0;
            var o_val = o_i[i] * rescale;
            
            for (var j = 0u; j < Bc; j = j + 1u) {
                let p_val = exp(s_p[row * Bc + j] - row_max);
                p_sum = p_sum + p_val;
                o_val = o_val + p_val * s_v[j * D + col];
            }
            
            m_i[i] = row_max;
            l_i[i] = l_i[i] * rescale + p_sum;
            o_i[i] = o_val;
        }
        workgroupBarrier();
    }
    
    // 3. Write final O to global VRAM
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid * 4u + i;
        let r = idx / D;
        let c = idx % D;
        let global_out_idx = (bh * params.seq_len + q_start + r) * D + c;
        if (q_start + r < params.seq_len) {
            out[global_out_idx] = o_i[i] / l_i[i];
        }
    }
}