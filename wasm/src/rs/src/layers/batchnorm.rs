use wasm_bindgen::prelude::*;

use crate::{Tensor, DTypes};

#[wasm_bindgen(js_name = handleBatchNorm)]
pub fn handle_batchnorm(input: &Tensor, scale: &Tensor, bias: &Tensor, mean: &Tensor, variance: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let scale = if let DTypes::F32(arr) = &scale.get_data() { arr } else { panic!("Scale's type is not float32!"); };
            let bias = if let DTypes::F32(arr) = &bias.get_data() { arr } else { panic!("Bias's type is not float32!"); };
            let mean = if let DTypes::F32(arr) = &mean.get_data() { arr } else { panic!("Mean's type is not float32!"); };
            let variance = if let DTypes::F32(arr) = &variance.get_data() { arr } else { panic!("Variance's type is not float32!"); };

            let mut out = Vec::with_capacity(input.get_length());
            for (index, val) in arr.iter().enumerate() {
                out[index] = ((val - mean[index]) / variance[index].sqrt()) * scale[index] + bias[index];
            }

            DTypes::F32(out)
        },
        DTypes::F64(arr) => {
            let scale = if let DTypes::F64(arr) = &scale.get_data() { arr } else { panic!("Scale's type is not float32!"); };
            let bias = if let DTypes::F64(arr) = &bias.get_data() { arr } else { panic!("Bias's type is not float32!"); };
            let mean = if let DTypes::F64(arr) = &mean.get_data() { arr } else { panic!("Mean's type is not float32!"); };
            let variance = if let DTypes::F64(arr) = &variance.get_data() { arr } else { panic!("Variance's type is not float32!"); };

            let mut out = Vec::with_capacity(input.get_length());
            for (index, val) in arr.iter().enumerate() {
                out[index] = ((val - mean[index]) / variance[index].sqrt()) * scale[index] + bias[index];
            }

            DTypes::F64(out)
        },
        _ => panic!("Batch normalization not support on target data type!")
    };

    Tensor::new(out_data, out_shape)
}