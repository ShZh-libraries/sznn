use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor};

#[wasm_bindgen(js_name = handleMaxPool2D)]
pub fn handle_maxpool_2d(
    input: &Tensor,
    kernel_height: usize,
    kernel_width: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    stride_y: usize,
    stride_x: usize,
) -> Tensor {
    let mut output = Tensor::new_empty();

    let in_shape = input.get_shape();
    let max_y = in_shape[2] + pad_top + pad_bottom - kernel_height;
    let max_x = in_shape[3] + pad_left + pad_right - kernel_width;
    let out_height = max_y / stride_y + 1;
    let out_width = max_x / stride_x + 1;
    let out_shape = vec![in_shape[0], in_shape[1], out_height, out_width];
    output.set_shape(out_shape);

    let in_channel_size = in_shape[2] * in_shape[3];
    let in_size = in_shape[1] * in_channel_size;

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in
                            (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x)
                        {
                            let max_idx = n as isize * in_size as isize
                                + c as isize * in_channel_size as isize
                                + y * in_shape[3] as isize
                                + x;
                            let mut max_val = arr[max_idx as usize];
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0
                                        && cy < in_shape[2] as isize
                                        && cx >= 0
                                        && cx < in_shape[3] as isize
                                    {
                                        let cur_idx = n * in_size
                                            + c * in_channel_size
                                            + cy as usize * in_shape[3]
                                            + cx as usize;
                                        arr[cur_idx]
                                    } else {
                                        0.
                                    };

                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                            out_data[out_idx] = max_val;
                            out_idx += 1;
                        }
                    }
                }
            }
            DTypes::F32(out_data)
        }
        DTypes::F64(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in
                            (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x)
                        {
                            let max_idx = n as isize * in_size as isize
                                + c as isize * in_channel_size as isize
                                + y * in_shape[3] as isize
                                + x;
                            let mut max_val = arr[max_idx as usize];
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0
                                        && cy < in_shape[2] as isize
                                        && cx >= 0
                                        && cx < in_shape[3] as isize
                                    {
                                        let cur_idx = n * in_size
                                            + c * in_channel_size
                                            + cy as usize * in_shape[3]
                                            + cx as usize;
                                        arr[cur_idx]
                                    } else {
                                        0.
                                    };

                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                            out_data[out_idx] = max_val;
                            out_idx += 1;
                        }
                    }
                }
            }
            DTypes::F64(out_data)
        }
        _ => panic!("Max pooling does not support these data types."),
    };
    output.set_data(out_data);

    output
}

#[wasm_bindgen(js_name = handleAvgPool2D)]
pub fn handle_avgpool_2d(
    input: &Tensor,
    kernel_height: usize,
    kernel_width: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    stride_y: usize,
    stride_x: usize,
) -> Tensor {
    let mut output = Tensor::new_empty();

    let in_shape = input.get_shape();
    let max_y = in_shape[2] + pad_top + pad_bottom - kernel_height;
    let max_x = in_shape[3] + pad_left + pad_right - kernel_width;
    let out_height = max_y / stride_y + 1;
    let out_width = max_x / stride_x + 1;
    let out_shape = vec![in_shape[0], in_shape[1], out_height, out_width];
    output.set_shape(out_shape);

    let in_channel_size = in_shape[2] * in_shape[3];
    let in_size = in_shape[1] * in_channel_size;
    let kernel_size = kernel_height * kernel_width;

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in
                            (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x)
                        {
                            let mut sum = 0.;
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0
                                        && cy < in_shape[2] as isize
                                        && cx >= 0
                                        && cx < in_shape[3] as isize
                                    {
                                        let cur_idx = n * in_size
                                            + c * in_channel_size
                                            + cy as usize * in_shape[3]
                                            + cx as usize;
                                        arr[cur_idx]
                                    } else {
                                        0.
                                    };

                                    sum += val;
                                }
                            }
                            out_data[out_idx] = sum / kernel_size as f32;
                            out_idx += 1;
                        }
                    }
                }
            }
            DTypes::F32(out_data)
        }
        DTypes::F64(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in
                            (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x)
                        {
                            let mut sum = 0.;
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0
                                        && cy < in_shape[2] as isize
                                        && cx >= 0
                                        && cx < in_shape[3] as isize
                                    {
                                        let cur_idx = n * in_size
                                            + c * in_channel_size
                                            + cy as usize * in_shape[3]
                                            + cx as usize;
                                        arr[cur_idx]
                                    } else {
                                        0.
                                    };

                                    sum += val;
                                }
                            }
                            out_data[out_idx] = sum / kernel_size as f64;
                            out_idx += 1;
                        }
                    }
                }
            }
            DTypes::F64(out_data)
        }
        _ => panic!("Max pooling does not support these data types."),
    };
    output.set_data(out_data);

    output
}

#[wasm_bindgen(js_name = handleGlobalAvgPool)]
pub fn handle_global_avgpool(input: &Tensor) -> Tensor {
    let in_shape = input.get_shape();
    handle_avgpool_2d(input, in_shape[2], in_shape[3], 0, 0, 0, 0, 1, 1)
}
