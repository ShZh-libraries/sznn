use std::sync::Mutex;

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor};
use rayon::prelude::*;

use super::extract_data;

macro_rules! instancenorm {
    (
        $arr: expr, $weight: expr, $bias: expr, $epsilon: expr,
        $in_shape: expr, $channel_size: expr, $typ: ty, $enu: path
    ) => {{
        let mut out = vec![0.; $arr.len()];
        let out_mutex = Mutex::new(&mut out);
        let weight = extract_data!($weight, $enu);
        let bias = extract_data!($bias, $enu);

        (0..$in_shape[1]).into_par_iter().for_each(|c| {
            let offset = c * $channel_size;
            // Calculate sum
            let sum: $typ = $arr[offset..(offset + $channel_size)].into_par_iter().sum();
            let mean = sum / $channel_size as $typ;
            // Calculate variance
            let mut variance = 0.;
            $arr[offset..(offset + $channel_size)]
                .into_iter()
                .for_each(|item| {
                    let std_var = item - mean;
                    variance += std_var * std_var;
                });
            variance /= $channel_size as $typ;
            // Normalization
            for i in 0..$channel_size {
                let mut out = out_mutex.lock().unwrap();
                out[offset + i] = weight[c] * ($arr[offset + i] - mean)
                    / (variance + $epsilon as $typ).sqrt()
                    + bias[c];
            }
        });

        out
    }};
}

#[wasm_bindgen(js_name = handleInstanceNorm)]
pub fn handle_instancenorm(input: &Tensor, weight: &Tensor, bias: &Tensor, epsilon: f64) -> Tensor {
    let in_shape = input.get_shape();
    let channel_size = in_shape[2] * in_shape[3];

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => DTypes::F32(instancenorm!(
            arr,
            weight,
            bias,
            epsilon,
            in_shape,
            channel_size,
            f32,
            DTypes::F32
        )),
        DTypes::F64(arr) => DTypes::F64(instancenorm!(
            arr,
            weight,
            bias,
            epsilon,
            in_shape,
            channel_size,
            f64,
            DTypes::F64
        )),
        _ => panic!("Convolutional does not support these data type!!"),
    };

    Tensor::new(out_data, in_shape)
}
