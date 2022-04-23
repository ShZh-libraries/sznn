use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};
use std::arch::wasm32::*;

#[wasm_bindgen(js_name = handleRelu)]
pub fn handle_relu(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => {
            let mut out: Vec<i8> = vec![0; arr.len()];
            let zeros = i8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            arr.par_chunks(16)
                .zip(out.par_chunks_mut(16))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let relu = i8x16_max(src, zeros);
                    v128_store(dst.as_mut_ptr() as *mut v128, relu);
                });
            DTypes::I8(out)
        }
        DTypes::I16(arr) => {
            let mut out: Vec<i16> = vec![0; arr.len()];
            let zeros = i16x8(0, 0, 0, 0, 0, 0, 0, 0);
            arr.par_chunks(8)
                .zip(out.par_chunks_mut(8))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let relu = i16x8_max(src, zeros);
                    v128_store(dst.as_mut_ptr() as *mut v128, relu);
                });
            DTypes::I16(out)
        }
        DTypes::I32(arr) => {
            let mut out: Vec<i32> = vec![0; arr.len()];
            let zeros = i32x4(0, 0, 0, 0);
            arr.par_chunks(4)
                .zip(out.par_chunks_mut(4))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let relu = i32x4_max(src, zeros);
                    v128_store(dst.as_mut_ptr() as *mut v128, relu);
                });
            DTypes::I32(out)
        }
        DTypes::F32(arr) => {
            let mut out: Vec<f32> = vec![0.; arr.len()];
            let zeros = f32x4(0., 0., 0., 0.);
            arr.par_chunks(4)
                .zip(out.par_chunks_mut(4))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let relu = f32x4_pmax(src, zeros);
                    v128_store(dst.as_mut_ptr() as *mut v128, relu);
                });
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let mut out: Vec<f64> = vec![0.; arr.len()];
            let zeros = f64x2(0., 0.);
            arr.par_chunks(2)
                .zip(out.par_chunks_mut(2))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let relu = f64x2_pmax(src, zeros);
                    v128_store(dst.as_mut_ptr() as *mut v128, relu);
                });
            DTypes::F64(out)
        }
        _ => panic!("Data type not supported in relu layer!"),
    };

    Tensor::new(out_data, out_shape)
}

#[wasm_bindgen(js_name = handleLeakyRelu)]
pub fn handle_leaky_relu(input: &Tensor, alpha: f64) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let out = arr
                .par_iter()
                .map(|&x| if x > 0. { x } else { alpha as f32 * x })
                .collect::<Vec<_>>();
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let out = arr
                .par_iter()
                .map(|&x| if x > 0. { x } else { alpha * x })
                .collect::<Vec<_>>();
            DTypes::F64(out)
        }
        _ => panic!("Data type not supported in leaky relu layer!"),
    };

    Tensor::new(out_data, out_shape)
}
