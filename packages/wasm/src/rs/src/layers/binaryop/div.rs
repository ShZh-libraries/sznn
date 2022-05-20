use paste::paste;
use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{layers::extract_data, DTypes, Tensor};
use std::{arch::wasm32::*, mem, sync::Mutex};

use super::*;

fn div_same_shape(a: &Tensor, b: &Tensor, out_shape: Vec<usize>) -> Tensor {
    let out_data = match &a.get_data() {
        DTypes::I8(a) => {
            let b = extract_data!(b, DTypes::I8);
            DTypes::I8(par!(a, b, i8))
        }
        DTypes::I16(a) => {
            let b = extract_data!(b, DTypes::I16);
            DTypes::I16(par!(a, b, i16))
        }
        DTypes::I32(a) => {
            let b = extract_data!(b, DTypes::I32);
            DTypes::I32(par!(a, b, i32))
        }
        DTypes::U8(a) => {
            let b = extract_data!(b, DTypes::U8);
            DTypes::U8(par!(a, b, u8))
        }
        DTypes::U16(a) => {
            let b = extract_data!(b, DTypes::U16);
            DTypes::U16(par!(a, b, u16))
        }
        DTypes::U32(a) => {
            let b = extract_data!(b, DTypes::U32);
            DTypes::U32(par!(a, b, u32))
        }
        DTypes::F32(a) => {
            let b = extract_data!(b, DTypes::F32);
            DTypes::F32(simd_par!(a, b, f32, f32x4_div))
        }
        DTypes::F64(a) => {
            let b = extract_data!(b, DTypes::F64);
            DTypes::F64(simd_par!(a, b, f64, f64x2_div))
        }
    };

    Tensor::new(out_data, out_shape)
}

broadcast!(div, /);

handle_binaryop!(div, /);
