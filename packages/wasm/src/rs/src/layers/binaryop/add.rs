use paste::paste;
use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{layers::extract_data, DTypes, Tensor};
use std::{arch::wasm32::*, mem, sync::Mutex};

use super::*;

fn add_same_shape(a: &Tensor, b: &Tensor, out_shape: Vec<usize>) -> Tensor {
    let out_data = match &a.get_data() {
        DTypes::I8(a) => {
            let b = extract_data!(b, DTypes::I8);
            DTypes::I8(simd_par!(a, b, i8, i8x16_add))
        }
        DTypes::I16(a) => {
            let b = extract_data!(b, DTypes::I16);
            DTypes::I16(simd_par!(a, b, i16, i16x8_add))
        }
        DTypes::I32(a) => {
            let b = extract_data!(b, DTypes::I32);
            DTypes::I32(simd_par!(a, b, i32, i32x4_add))
        }
        DTypes::U8(a) => {
            let b = extract_data!(b, DTypes::U8);
            DTypes::U8(simd_par!(a, b, u8, u8x16_add))
        }
        DTypes::U16(a) => {
            let b = extract_data!(b, DTypes::U16);
            DTypes::U16(simd_par!(a, b, u16, u16x8_add))
        }
        DTypes::U32(a) => {
            let b = extract_data!(b, DTypes::U32);
            DTypes::U32(simd_par!(a, b, u32, u32x4_add))
        }
        DTypes::F32(a) => {
            let b = extract_data!(b, DTypes::F32);
            DTypes::F32(simd_par!(a, b, f32, f32x4_add))
        }
        DTypes::F64(a) => {
            let b = extract_data!(b, DTypes::F64);
            DTypes::F64(simd_par!(a, b, f64, f64x2_add))
        }
    };

    Tensor::new(out_data, out_shape)
}

broadcast!(add, +);

handle_binaryop!(add, +);
