use wasm_bindgen::prelude::wasm_bindgen;

use crate::{layers::padding::handle_padding, DTypes, Tensor};

macro_rules! maxpool {
    (
        $arr: expr, $in_shape: expr, $out_shape: expr, $len: expr,
        $kernel_height: expr, $kernel_width: expr,
        $stride_y: expr, $stride_x: expr, $typ: ty
    ) => {{
        let in_channel_size = $in_shape[2] * $in_shape[3];
        let row_stride = $in_shape[3] - $kernel_width;

        let mut out_idx = 0;
        let mut out_data = vec![0.; $len];
        for c in 0..$out_shape[1] {
            for y in 0..$out_shape[2] {
                for x in 0..$out_shape[3] {
                    let in_start_y = y * $stride_y;
                    let in_start_x = x * $stride_x;

                    let mut offset = c * in_channel_size + in_start_y * $in_shape[3] + in_start_x;
                    let mut max = $arr[offset];
                    for _ky in 0..$kernel_height {
                        for _kx in 0..$kernel_width {
                            let val = $arr[offset];
                            max = max.max(val);

                            offset += 1;
                        }
                        offset += row_stride;
                    }
                    out_data[out_idx] = max as $typ;
                    out_idx += 1;
                }
            }
        }

        out_data
    }};
}

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
    let padding = if pad_top != 0 || pad_left != 0 || pad_bottom != 0 || pad_right != 0 {
        handle_padding(input, pad_top, pad_left, pad_bottom, pad_right)
    } else {
        input.clone()
    };

    let mut output = Tensor::new_empty();

    let in_shape = padding.get_shape();
    let out_shape = vec![
        in_shape[0],
        in_shape[1],
        (in_shape[2] - kernel_height) / stride_y + 1,
        (in_shape[3] - kernel_width) / stride_x + 1,
    ];
    output.set_shape(out_shape.clone());

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => DTypes::F32(maxpool!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            kernel_height,
            kernel_width,
            stride_y,
            stride_x,
            f32
        )),
        DTypes::F64(arr) => DTypes::F64(maxpool!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            kernel_height,
            kernel_width,
            stride_y,
            stride_x,
            f64
        )),
        _ => panic!("Max pooling does not support these data types."),
    };
    output.set_data(out_data);

    output
}

macro_rules! avgpool {
    (
        $arr: expr, $in_shape: expr, $out_shape: expr, $len: expr,
        $kernel_height: expr, $kernel_width: expr,
        $stride_y: expr, $stride_x: expr, $typ: ty
    ) => {{
        let in_channel_size = $in_shape[2] * $in_shape[3];
        let kernel_size = $kernel_height * $kernel_width;
        let row_stride = $in_shape[3] - $kernel_width;

        let mut out_idx = 0;
        let mut out_data = vec![0.; $len];
        for _ in 0..$out_shape[0] {
            for c in 0..$out_shape[1] {
                for y in 0..$out_shape[2] {
                    for x in 0..$out_shape[3] {
                        let in_start_y = y * $stride_y;
                        let in_start_x = x * $stride_x;

                        let mut sum = 0.;
                        let mut offset =
                            c * in_channel_size + in_start_y * $in_shape[3] + in_start_x;
                        for _ in 0..$kernel_height {
                            for _ in 0..$kernel_width {
                                let val = $arr[offset];

                                sum += val;
                                offset += 1;
                            }
                            offset += row_stride;
                        }
                        out_data[out_idx] = sum / kernel_size as $typ;
                        out_idx += 1;
                    }
                }
            }
        }

        out_data
    }};
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
    let padding = if pad_top != 0 || pad_left != 0 || pad_bottom != 0 || pad_right != 0 {
        handle_padding(input, pad_top, pad_left, pad_bottom, pad_right)
    } else {
        input.clone()
    };

    let mut output = Tensor::new_empty();

    let in_shape = padding.get_shape();
    let out_shape = vec![
        in_shape[0],
        in_shape[1],
        (in_shape[2] - kernel_height) / stride_y + 1,
        (in_shape[3] - kernel_width) / stride_x + 1,
    ];
    output.set_shape(out_shape.clone());

    let out_data = match &padding.get_data() {
        DTypes::F32(arr) => DTypes::F32(avgpool!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            kernel_height,
            kernel_width,
            stride_y,
            stride_x,
            f32
        )),
        DTypes::F64(arr) => DTypes::F64(avgpool!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            kernel_height,
            kernel_width,
            stride_y,
            stride_x,
            f64
        )),
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
