use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};
use std::arch::wasm32::*;

#[wasm_bindgen(js_name = handleFloor)]
pub fn handle_floor(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let mut out: Vec<f32> = vec![0.; arr.len()];
            arr.par_chunks(4)
                .zip(out.par_chunks_mut(4))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let result = f32x4_floor(src);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let mut out: Vec<f64> = vec![0.; arr.len()];
            arr.par_chunks(2)
                .zip(out.par_chunks_mut(2))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let result = f64x2_floor(src);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            DTypes::F64(out)
        }
        _ => panic!("Data type not supported in floor layer!"),
    };

    Tensor::new(out_data, out_shape)
}
