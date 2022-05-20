use paste::paste;
use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};
use std::{arch::wasm32::*, mem};

macro_rules! par {
    ($arr: expr, $fn: ident) => {
        $arr.par_iter().map(|x| x.$fn()).collect::<Vec<_>>()
    };
}

macro_rules! simd_par {
    ($arr: expr, $typ: ty, $simd: ident) => {{
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
    }};
}

macro_rules! handle_unary_par_inner {
    ($name: ident, $op: ident) => {
        paste! {
            #[wasm_bindgen(js_name = [< handle $name:camel >])]
            pub fn [< handle $name >] (input: &Tensor) -> Tensor {
                let out_shape = input.get_shape();
                let out_data = match &input.get_data() {
                    DTypes::F32(arr) => DTypes::F32(par!(arr, $op)),
                    DTypes::F64(arr) => DTypes::F64(par!(arr, $op)),
                    _ => {
                        panic!("Operation not supported on target type!");
                    }
                };

                Tensor::new(out_data, out_shape)
            }
        }
    };
}

macro_rules! handle_unary_simd_float_inner {
    ($name: ident, $op: ident) => {
        paste! {
            #[wasm_bindgen(js_name = [< handle $name:camel >] )]
            pub fn [< handle $name >] (input: &Tensor) -> Tensor {
                let out_shape = input.get_shape();
                let out_data = match &input.get_data() {
                    DTypes::F32(arr) => DTypes::F32(simd_par!(arr, f32, [< f32x4_ $op >])),
                    DTypes::F64(arr) => DTypes::F64(simd_par!(arr, f64, [< f64x2_ $op >])),
                    _ => panic!("Data type not supported in this layer!"),
                };

                Tensor::new(out_data, out_shape)
            }
        }
    };
}

macro_rules! handle_unary_simd_inner {
    ($name: ident, $op: ident) => {
        paste! {
            #[wasm_bindgen(js_name = [< handle $name:camel >] )]
            pub fn [< handle $name >] (input: &Tensor) -> Tensor {
                let out_shape = input.get_shape();
                let out_data = match &input.get_data() {
                    DTypes::I8(arr) => DTypes::I8(simd_par!(arr, i8, [< i8x16_ $op >])),
                    DTypes::I16(arr) => DTypes::I16(simd_par!(arr, i16, [< i16x8_ $op >])),
                    DTypes::I32(arr) => DTypes::I32(simd_par!(arr, i32, [< i32x4_ $op >])),
                    DTypes::F32(arr) => DTypes::F32(simd_par!(arr, f32, [< f32x4_ $op >])),
                    DTypes::F64(arr) => DTypes::F64(simd_par!(arr, f64, [< f64x2_ $op >])),
                    _ => panic!("Data type not supported in this layer!"),
                };

                Tensor::new(out_data, out_shape)
            }
        }
    };
}

macro_rules! handle_unary_par {
    ($unary: ident) => {
        handle_unary_par_inner!($unary, $unary);
    };
    ($name: ident, $op: ident) => {
        handle_unary_par_inner!($name, $op);
    };
}

macro_rules! handle_unary_simd_float {
    ($unary: ident) => {
        handle_unary_simd_float_inner!($unary, $unary);
    };
    ($name: ident, $op: ident) => {
        handle_unary_simd_float_inner!($name, $op);
    };
}

macro_rules! handle_unary_simd {
    ($unary: ident) => {
        handle_unary_simd_inner!($unary, $unary);
    };
    ($name: ident, $op: ident) => {
        handle_unary_simd_inner!($name, $op);
    };
}

handle_unary_simd!(abs);
handle_unary_par!(acos);
handle_unary_par!(acosh);
handle_unary_par!(asin);
handle_unary_par!(asinh);
handle_unary_par!(atan);
handle_unary_par!(atanh);
handle_unary_simd_float!(ceil);
handle_unary_par!(cos);
handle_unary_par!(cosh);
handle_unary_simd_float!(floor);
handle_unary_par!(log, ln);
handle_unary_simd!(neg);
handle_unary_simd_float!(round, nearest);
handle_unary_par!(sign, signum);
handle_unary_par!(sin);
handle_unary_par!(sinh);
handle_unary_simd_float!(sqrt);
handle_unary_par!(tan);
handle_unary_par!(tanh);

#[wasm_bindgen(js_name = handleIdentity)]
pub fn handle_identity(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = input.get_data();

    Tensor::new(out_data.clone(), out_shape)
}

#[wasm_bindgen(js_name = handleSigmoid)]
pub fn handle_sigmoid(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let out = arr
                .par_iter()
                .map(|x| 1. / (1. + x.exp()))
                .collect::<Vec<_>>();
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let out = arr
                .par_iter()
                .map(|x| 1. / (1. + x.exp()))
                .collect::<Vec<_>>();
            DTypes::F64(out)
        }
        _ => {
            panic!("Operation not supported on target type!");
        }
    };

    Tensor::new(out_data, out_shape)
}
