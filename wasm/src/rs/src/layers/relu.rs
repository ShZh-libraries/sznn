use wasm_bindgen::prelude::*;

use crate::{Tensor, TensorDataType};

#[wasm_bindgen]
pub fn forward_relu(input: &Tensor) -> Tensor {
    let mut output = Tensor::new();
    output.set_vec_shape(input.get_shape());
    output.data = match &input.data {
        TensorDataType::Int8(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            TensorDataType::Int8(out)
        },
        TensorDataType::Int16(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            TensorDataType::Int16(out)
        },
        TensorDataType::Int32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0 { x } else { 0 })
                .collect::<Vec<_>>();
            TensorDataType::Int32(out)
        },
        TensorDataType::Float32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { 0. })
                .collect::<Vec<_>>();
            TensorDataType::Float32(out)
        },
        TensorDataType::Float64(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { 0. })
                .collect::<Vec<_>>();
            TensorDataType::Float64(out)
        },
        _ => panic!("ReLu operation only support float32 and float64 dtype!")
    };

    output
}

#[wasm_bindgen]
pub fn forward_leaky_relu(input: &Tensor, alpha: f64) -> Tensor {
    let mut output = Tensor::new();
    output.set_vec_shape(input.get_shape());
    output.data = match &input.data {
        TensorDataType::Float32(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { alpha as f32 * x })
                .collect::<Vec<_>>();
            TensorDataType::Float32(out)
        },
        TensorDataType::Float64(arr) => {
            let out = arr
                .iter()
                .map(|&x| if x > 0. { x } else { alpha * x })
                .collect::<Vec<_>>();
            TensorDataType::Float64(out)
        },
        _ => panic!("ReLu operation only support float32 and float64 dtype!")
    };

    output
}
