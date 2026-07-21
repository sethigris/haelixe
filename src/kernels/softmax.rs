use crate::{Tensor, DType};
use rayon::prelude::*;

pub fn softmax_forward(x: &Tensor) -> Tensor {
    let x_cpu = x.ensure_cpu();
    
    // DEBUG: Let's prove this kernel is actually being executed!

    let dims = x_cpu.shape.dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = x_cpu.shape.num_elements() / last_dim;

    let mut out_data = vec![0.0f32; x_cpu.shape.num_elements()];
    let out_addr = out_data.as_mut_ptr() as usize;

    // Match the pointer type to the actual underlying memory layout
    match x_cpu.dtype {
        DType::F32 => {
            let x_addr = x_cpu.storage.as_ptr() as *const f32 as usize;
            (0..batch_size).into_par_iter().for_each(|b| {
                unsafe {
                    let row_start = b * last_dim;
                    let x_row = (x_addr as *const f32).add(row_start);
                    let out_row = (out_addr as *mut f32).add(row_start);

                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..last_dim { let val = *x_row.add(i); if val > max_val { max_val = val; } }

                    let mut sum_exp = 0.0f32;
                    for i in 0..last_dim { let val = (*x_row.add(i) - max_val).exp(); *out_row.add(i) = val; sum_exp += val; }

                    let inv_sum = 1.0 / sum_exp;
                    for i in 0..last_dim { *out_row.add(i) *= inv_sum; }
                }
            });
        }
        DType::F64 => {
            // The F64 branch safely reads 64-bit floats and casts them down to 32-bit for the output
            let x_addr = x_cpu.storage.as_ptr() as *const f64 as usize;
            (0..batch_size).into_par_iter().for_each(|b| {
                unsafe {
                    let row_start = b * last_dim;
                    let x_row = (x_addr as *const f64).add(row_start);
                    let out_row = (out_addr as *mut f32).add(row_start);

                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..last_dim { let val = *x_row.add(i); if val > max_val { max_val = val; } }

                    let mut sum_exp = 0.0f64;
                    for i in 0..last_dim { 
                        let val = (*x_row.add(i) - max_val).exp(); 
                        *out_row.add(i) = val as f32; 
                        sum_exp += val; 
                    }

                    let inv_sum = (1.0 / sum_exp) as f32;
                    for i in 0..last_dim { *out_row.add(i) *= inv_sum; }
                }
            });
        }
        _ => panic!("Softmax forward pass only supports F32 and F64")
    }
    Tensor::from_slice(DType::F32, x_cpu.shape.clone(), &out_data)
}

pub fn softmax_backward(y: &Tensor, grad: &Tensor) -> Tensor {
    let y_cpu = y.ensure_cpu();
    let g_cpu = grad.ensure_cpu();
    

    let dims = y_cpu.shape.dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = y_cpu.shape.num_elements() / last_dim;

    // y is guaranteed F32 because we forced it in the forward pass
    let y_addr = y_cpu.storage.as_ptr() as *const f32 as usize;
    let mut dx_data = vec![0.0f32; y_cpu.shape.num_elements()];
    let dx_addr = dx_data.as_mut_ptr() as usize;

    match g_cpu.dtype {
        DType::F32 => {
            let g_addr = g_cpu.storage.as_ptr() as *const f32 as usize;
            (0..batch_size).into_par_iter().for_each(|b| {
                unsafe {
                    let offset = b * last_dim;
                    let y_row = (y_addr as *const f32).add(offset);
                    let g_row = (g_addr as *const f32).add(offset);
                    let dx_row = (dx_addr as *mut f32).add(offset);

                    let mut dot = 0.0f32;
                    for i in 0..last_dim { dot += *g_row.add(i) * *y_row.add(i); }
                    for i in 0..last_dim { *dx_row.add(i) = *y_row.add(i) * (*g_row.add(i) - dot); }
                }
            });
        }
        DType::F64 => {
            let g_addr = g_cpu.storage.as_ptr() as *const f64 as usize;
            (0..batch_size).into_par_iter().for_each(|b| {
                unsafe {
                    let offset = b * last_dim;
                    let y_row = (y_addr as *const f32).add(offset);
                    let g_row = (g_addr as *const f64).add(offset);
                    let dx_row = (dx_addr as *mut f32).add(offset);

                    let mut dot = 0.0f64;
                    for i in 0..last_dim { dot += *g_row.add(i) * (*y_row.add(i) as f64); }
                    let dot_f32 = dot as f32;
                    
                    for i in 0..last_dim { 
                        *dx_row.add(i) = *y_row.add(i) * ((*g_row.add(i) as f32) - dot_f32); 
                    }
                }
            });
        }
        _ => panic!("Softmax backward pass only supports F32 and F64 grads")
    }
    Tensor::from_slice(DType::F32, y_cpu.shape.clone(), &dx_data)
}