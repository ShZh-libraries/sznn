use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};

use super::extract_data;

macro_rules! batchnorm {
    ($input: expr, $scale: expr, $bias: expr, $mean: expr, $variance: expr, $dtype: path) => {{
        let scale = extract_data!(&$scale, $dtype);
        let bias = extract_data!(&$bias, $dtype);
        let mean = extract_data!(&$mean, $dtype);
        let variance = extract_data!(&$variance, $dtype);

        let mut out = Vec::with_capacity($input.len());
        for (index, val) in $input.iter().enumerate() {
            out[index] =
                ((val - mean[index]) / variance[index].sqrt()) * scale[index] + bias[index];
        }

        out
    }};
}

#[wasm_bindgen(js_name = handleBatchNorm)]
pub fn handle_batchnorm(
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    variance: &Tensor,
) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => DTypes::F32(batchnorm!(arr, scale, bias, mean, variance, DTypes::F32)),
        DTypes::F64(arr) => DTypes::F64(batchnorm!(arr, scale, bias, mean, variance, DTypes::F64)),
        _ => panic!("Batch normalization not support on target data type!"),
    };

    Tensor::new(out_data, out_shape)
}
