use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::tensor::{DTypes, Tensor};

macro_rules! cast_to_type {
    ($arr: expr, $typ: ty) => {{
        let mut out = vec![0 as $typ; $arr.len()];
        $arr.par_iter()
            .map(|x| *x as $typ)
            .collect_into_vec(&mut out);
        out
    }};
}

macro_rules! cast {
    ($arr: expr, $to: expr) => {{
        match $to {
            1 => DTypes::F32(cast_to_type!($arr, f32)),
            2 => DTypes::U8(cast_to_type!($arr, u8)),
            3 => DTypes::I8(cast_to_type!($arr, i8)),
            4 => DTypes::U16(cast_to_type!($arr, u16)),
            5 => DTypes::I16(cast_to_type!($arr, i16)),
            6 => DTypes::I32(cast_to_type!($arr, i32)),
            11 => DTypes::F64(cast_to_type!($arr, f64)),
            _ => {
                panic!("Type not support!");
            }
        }
    }};
}

#[wasm_bindgen(js_name = handleCast)]
pub fn handle_cast(input: &Tensor, to: usize) -> Tensor {
    let out_shape = input.get_shape();

    let out_data = match &input.get_data() {
        DTypes::I8(arr) => cast!(arr, to),
        DTypes::I16(arr) => cast!(arr, to),
        DTypes::I32(arr) => cast!(arr, to),
        DTypes::U8(arr) => cast!(arr, to),
        DTypes::U16(arr) => cast!(arr, to),
        DTypes::U32(arr) => cast!(arr, to),
        DTypes::F32(arr) => cast!(arr, to),
        DTypes::F64(arr) => cast!(arr, to),
    };

    Tensor::new(out_data, out_shape)
}
