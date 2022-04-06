use wasm_bindgen::prelude::*;

use crate::{Tensor, DTypes};

#[wasm_bindgen(js_name = handleRelu)]
pub fn handle_relu(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            DTypes::I8(out)
        },
        DTypes::I16(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            DTypes::I16(out)
        },
        DTypes::I32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            DTypes::I32(out)
        },
        DTypes::F32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { 0. })
                .collect::<Vec<_>>();
            DTypes::F32(out)
        },
        DTypes::F64(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { 0. })
                .collect::<Vec<_>>();
            DTypes::F64(out)
        },
        _ => panic!("Data type not supported in relu layer!")
    };

    Tensor::new(out_data, out_shape)
}

#[wasm_bindgen(js_name = handleLeakyRelu)]
pub fn handle_leaky_relu(input: &Tensor, alpha: f64) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { alpha as f32 * x })
                .collect::<Vec<_>>();
            DTypes::F32(out)
        },
        DTypes::F64(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { alpha * x })
                .collect::<Vec<_>>();
            DTypes::F64(out)
        },
        _ => panic!("Data type not supported in leaky relu layer!")
    };

    Tensor::new(out_data, out_shape)
}
