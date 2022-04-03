use wasm_bindgen::prelude::*;

use crate::{Tensor, TensorDataType};

#[wasm_bindgen]
pub fn forward(input: &Tensor, scale: &Tensor, bias: &Tensor, mean: &Tensor, variance: &Tensor) -> Tensor {
    let mut output = Tensor::new();

    output.set_vec_shape(input.get_shape());
    output.data = match &input.data {
        TensorDataType::Float32(arr) => {
            let scale = if let TensorDataType::Float32(arr) = &scale.data { arr } else { panic!("Scale's type is not float32!"); };
            let bias = if let TensorDataType::Float32(arr) = &bias.data { arr } else { panic!("Bias's type is not float32!"); };
            let mean = if let TensorDataType::Float32(arr) = &mean.data { arr } else { panic!("Mean's type is not float32!"); };
            let variance = if let TensorDataType::Float32(arr) = &variance.data { arr } else { panic!("Variance's type is not float32!"); };

            let mut out = Vec::with_capacity(input.get_length());
            for (index, val) in arr.iter().enumerate() {
                out[index] = ((val - mean[index]) / variance[index].sqrt()) * scale[index] + bias[index];
            }

            TensorDataType::Float32(out)
        },
        TensorDataType::Float64(arr) => {
            let scale = if let TensorDataType::Float64(arr) = &scale.data { arr } else { panic!("Scale's type is not float32!"); };
            let bias = if let TensorDataType::Float64(arr) = &bias.data { arr } else { panic!("Bias's type is not float32!"); };
            let mean = if let TensorDataType::Float64(arr) = &mean.data { arr } else { panic!("Mean's type is not float32!"); };
            let variance = if let TensorDataType::Float64(arr) = &variance.data { arr } else { panic!("Variance's type is not float32!"); };

            let mut out = Vec::with_capacity(input.get_length());
            for (index, val) in arr.iter().enumerate() {
                out[index] = ((val - mean[index]) / variance[index].sqrt()) * scale[index] + bias[index];
            }

            TensorDataType::Float64(out)
        },
        _ => panic!("Batch normalization not support on target data type!")
    };

    output
}