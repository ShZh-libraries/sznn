use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};
use std::{arch::wasm32::*, mem};

macro_rules! abs {
    ($arr: expr, $typ: ty, $simd: ident) => {
        {
            let chunks_num: usize = 16 / mem::size_of::<$typ>();
            let mut out = vec![0 as $typ; $arr.len()];
            $arr.par_chunks(chunks_num)
                .zip(out.par_chunks_mut(chunks_num))
                .for_each(|(src, dst)| unsafe {
                    let src = v128_load(src.as_ptr() as *const v128);
                    let result = $simd(src);
                    v128_store(dst.as_mut_ptr() as *mut v128, result);
                });
            out
        }
    };
}

#[wasm_bindgen(js_name = handleAbs)]
pub fn handle_abs(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => DTypes::I8(abs!(arr, i8, i8x16_abs)),
        DTypes::I16(arr) => DTypes::I16(abs!(arr, i16, i16x8_abs)),
        DTypes::I32(arr) => DTypes::I32(abs!(arr, i32, i32x4_abs)),
        DTypes::F32(arr) => DTypes::F32(abs!(arr, f32, f32x4_abs)),
        DTypes::F64(arr) => DTypes::F64(abs!(arr, f64, f64x2_abs)),
        _ => panic!("Data type not supported in abs layer!"),
    };

    Tensor::new(out_data, out_shape)
}
