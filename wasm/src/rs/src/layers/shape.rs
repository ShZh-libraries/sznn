use wasm_bindgen::prelude::wasm_bindgen;

use crate::{Tensor, DTypes};

#[wasm_bindgen]
pub fn handle_shape(input: &Tensor) -> Tensor {
    let shape = input.get_shape();
    let out_shape = vec![shape.len()];
    let out_data = match &input.get_data() {
        DTypes::I8(_) => {
            let out = shape.iter().map(|&x| x as i8).collect();
            DTypes::I8(out)
        },
        DTypes::I16(_) => {
            let out = shape.iter().map(|&x| x as i16).collect();
            DTypes::I16(out)
        },
        DTypes::I32(_) => {
            let out = shape.iter().map(|&x| x as i32).collect();
            DTypes::I32(out)
        },
        DTypes::U8(_) => {
            let out = shape.iter().map(|&x| x as u8).collect();
            DTypes::U8(out)
        },
        DTypes::U16(_) => {
            let out = shape.iter().map(|&x| x as u16).collect();
            DTypes::U16(out)
        },
        DTypes::U32(_) => {
            let out = shape.iter().map(|&x| x as u32).collect();
            DTypes::U32(out)
        },
        DTypes::F32(_) => {
            let out = shape.iter().map(|&x| x as f32).collect();
            DTypes::F32(out)
        },
        DTypes::F64(_) => {
            let out = shape.iter().map(|&x| x as f64).collect();
            DTypes::F64(out)
        },
    };

    Tensor::new(out_data, out_shape)
}
