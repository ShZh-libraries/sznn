use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};
use std::{arch::wasm32::*, sync::Mutex};

use super::{get_broadcast_shape, get_broadcast_dims, idx_to_loc, get_broadcast_loc, loc_to_idx};

#[wasm_bindgen(js_name = handleSub)]
pub fn handle_sub(a: &Tensor, b: &Tensor) -> Tensor {
    let (a_shape, b_shape) = (a.get_shape(), b.get_shape());
    let out_shape = get_broadcast_shape(&a_shape, &b_shape);
    let (a_broadcast_dim, b_broadcast_dim) = (get_broadcast_dims(&a_shape, &out_shape), get_broadcast_dims(&b_shape, &out_shape));

    if a_broadcast_dim.len() + b_broadcast_dim.len() == 0 {
        handle_same_shape(a, b, out_shape)
    } else {
        handle_broadcast(a, b, &a_broadcast_dim, &b_broadcast_dim, out_shape)
    }
}

fn handle_same_shape(a: &Tensor, b: &Tensor, out_shape: Vec<usize>) -> Tensor {
    let len = out_shape.iter().fold(1, |res, val| res * val);

    let out_data = match &a.get_data() {
        DTypes::I8(a) => {
            let b = if let DTypes::I8(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<i8> = vec![0; len];
            a.par_chunks(16)
                .zip(b.par_chunks(16))
                .zip(out.par_chunks_mut(16))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = i8x16_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::I8(out)
        }
        DTypes::I16(a) => {
            let b = if let DTypes::I16(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<i16> = vec![0; len];
            a.par_chunks(8)
                .zip(b.par_chunks(8))
                .zip(out.par_chunks_mut(8))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = i16x8_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::I16(out)
        },
        DTypes::I32(a) => {
            let b = if let DTypes::I32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<i32> = vec![0; len];
            a.par_chunks(4)
                .zip(b.par_chunks(4))
                .zip(out.par_chunks_mut(4))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = i32x4_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::I32(out)
        },
        DTypes::U8(a) => {
            let b = if let DTypes::U8(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<u8> = vec![0; len];
            a.par_chunks(16)
                .zip(b.par_chunks(16))
                .zip(out.par_chunks_mut(16))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = u8x16_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::U8(out)
        },
        DTypes::U16(a) => {
            let b = if let DTypes::U16(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<u16> = vec![0; len];
            a.par_chunks(8)
                .zip(b.par_chunks(8))
                .zip(out.par_chunks_mut(8))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = u16x8_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::U16(out)
        },
        DTypes::U32(a) => {
            let b = if let DTypes::U32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out: Vec<u32> = vec![0; len];
            a.par_chunks(4)
                .zip(b.par_chunks(4))
                .zip(out.par_chunks_mut(4))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = u32x4_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::U32(out)
        },
        DTypes::F32(a) => {
            let b = if let DTypes::F32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out = vec![0.; len];
            a.par_chunks(4)
                .zip(b.par_chunks(4))
                .zip(out.par_chunks_mut(4))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = f32x4_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::F32(out)
        },
        DTypes::F64(a) => {
            let b = if let DTypes::F64(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };
            let mut out = vec![0.; len];
            a.par_chunks(4)
                .zip(b.par_chunks(2))
                .zip(out.par_chunks_mut(2))
                .for_each(|((a, b), dst)| unsafe {
                    let a = v128_load(a.as_ptr() as *const v128);
                    let b = v128_load(b.as_ptr() as *const v128);
                    let result = f64x2_sub(a, b);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::F64(out)
        },
    };

    Tensor::new(out_data, out_shape)
}

fn handle_broadcast(a: &Tensor, b: &Tensor, a_broadcast_dim: &Vec<usize>, b_broadcast_dim: &Vec<usize>, out_shape: Vec<usize>) -> Tensor {
    let mut output = Tensor::new_empty();
    output.set_shape(out_shape);

    let a_stride = a.get_stride();
    let a_dim = a_stride.len();
    let b_stride = b.get_stride();
    let b_dim = b_stride.len();
    let out_len = output.get_length();
    let out_stride = output.get_stride();

    let out_data = match &a.get_data() {
        DTypes::I8(a) => {
            let b = if let DTypes::I8(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::I8(out_data)
        }
        DTypes::I16(a) => {
            let b = if let DTypes::I16(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::I16(out_data)
        },
        DTypes::I32(a) => {
            let b = if let DTypes::I32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::I32(out_data)
        },
        DTypes::U8(a) => {
            let b = if let DTypes::U8(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::U8(out_data)
        },
        DTypes::U16(a) => {
            let b = if let DTypes::U16(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::U16(out_data)
        },
        DTypes::U32(a) => {
            let b = if let DTypes::U32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::U32(out_data)
        },
        DTypes::F32(a) => {
            let b = if let DTypes::F32(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0.; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::F32(out_data)
        },
        DTypes::F64(a) => {
            let b = if let DTypes::F64(data) = b.get_data() {
                data
            } else {
                panic!("Two tensors' data types do not match!")
            };

            let mut out_data = vec![0.; out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &out_stride);

                let a_loc = get_broadcast_loc(&out_loc, a_dim, &*a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &a_stride);

                let b_loc = get_broadcast_loc(&out_loc, b_dim, &*b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &b_stride);

                let value = a[a_idx] - b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            DTypes::F64(out_data)
        },
    };

    output.set_data(out_data);

    output
}