use rayon::prelude::*;
use wasm_bindgen::prelude::wasm_bindgen;
use std::sync::Mutex;

use crate::{DTypes, Tensor, layers::padding::handle_padding};

use super::extract_data;

macro_rules! conv {
    (
        $arr: expr, $weight: expr, $bias: expr, $len: expr,
        $in_shape: expr, $weight_shape: expr,  $out_shape: expr,
        $stride_y: expr, $stride_x: expr
    ) => {
        {
            let kernel_channel_size = $weight_shape[2] * $weight_shape[3];
            let kernel_size = $weight_shape[1] * kernel_channel_size;
            let in_row_stride = $in_shape[3] - $weight_shape[3];
            let in_channel_stride = $in_shape[2] * $in_shape[3] - $weight_shape[2] * $in_shape[3];

            let mut out_data = vec![0.; $len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..$out_shape[1]).into_par_iter().for_each(|c| {
                (0..$out_shape[2]).into_par_iter().for_each(|y| {
                    (0..$out_shape[3]).into_par_iter().for_each(|x| {
                        let start_y = y * $stride_y;
                        let start_x = x * $stride_x;

                        let mut sum = 0.;
                        let mut weight_offset = c * kernel_size;
                        let mut in_offset = start_y * $in_shape[3] + start_x;

                        for _kc in 0..$weight_shape[1] {
                            for _ky in 0..$weight_shape[2] {
                                for _kx in 0..$weight_shape[3] {
                                    sum += $weight[weight_offset] * $arr[in_offset];

                                    in_offset += 1;
                                    weight_offset += 1;
                                }
                                in_offset += in_row_stride;
                            }
                            in_offset += in_channel_stride;
                        }

                        let out_idx = c * $out_shape[2] * $out_shape[3] + y * $out_shape[3] + x;
                        let mut out_data = out_data_mutex.lock().unwrap();
                        out_data[out_idx] = sum + $bias[c];
                    })
                })
            });

            out_data
        }
    };
}

#[wasm_bindgen(js_name = handleConv)]
pub fn handle_conv(
    kernel_height: usize,
    kernel_width: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    stride_y: usize,
    stride_x: usize,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<Tensor>,
) -> Tensor {
    let padding = if pad_top != 0 || pad_left != 0 || pad_bottom != 0 || pad_right != 0 {
        handle_padding(input, pad_top, pad_left, pad_bottom, pad_right)
    } else {
        input.clone()
    };

    let mut output = Tensor::new_empty();

    let in_shape = padding.get_shape();
    let weight_shape = weight.get_shape();
    let out_shape = vec![
        in_shape[0], 
        weight_shape[0], 
        (in_shape[2] - kernel_height) / stride_y + 1, 
        (in_shape[3] - kernel_width) / stride_x + 1,
    ];
    output.set_shape(out_shape.clone());

    let out_data = match &padding.get_data() {
        DTypes::F32(arr) => {
            let len = output.get_length();
            let weight = extract_data!(weight, DTypes::F32);
            let bias = bias.map_or(vec![0.; len], |tensor| {
                extract_data!(tensor, DTypes::F32).to_vec()
            });

            DTypes::F32(conv!(arr, weight, bias, len, in_shape, weight_shape, out_shape, stride_y, stride_x))
        }
        DTypes::F64(arr) => {
            let len = output.get_length();
            let weight = extract_data!(weight, DTypes::F64);
            let bias = bias.map_or(vec![0.; len], |tensor| {
                extract_data!(tensor, DTypes::F64).to_vec()
            });

            DTypes::F64(conv!(arr, weight, bias, len, in_shape, weight_shape, out_shape, stride_y, stride_x))
        }
        _ => {
            panic!("Convolutional does not support these data type!!")
        }
    };

    output.set_data(out_data);

    output
}
