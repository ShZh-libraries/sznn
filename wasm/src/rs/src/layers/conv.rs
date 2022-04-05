use wasm_bindgen::prelude::wasm_bindgen;

use crate::{Tensor, DTypes};

#[wasm_bindgen]
pub fn handle_conv(    
    kernel_height: usize, kernel_width: usize,
    pad_top: usize, pad_left: usize, 
    pad_bottom: usize, pad_right: usize, 
    stride_y: usize, stride_x: usize,
    input: &Tensor, weight: &Tensor, bias: Option<Tensor>
) -> Tensor {
    let mut output = Tensor::new_empty();

    let in_shape = input.get_shape();
    let max_y = in_shape[2] + pad_top + pad_bottom - kernel_height;
    let max_x = in_shape[3] + pad_left + pad_right - kernel_width;
    let out_height = max_y / stride_y + 1;
    let out_width = max_x / stride_x + 1;
    let out_shape = vec![in_shape[0], in_shape[1], out_height, out_width];
    output.set_shape(out_shape);

    let weight_shape = weight.get_shape();
    let kernel_channel_size = weight_shape[2] * weight_shape[3];
    let kernel_size = weight_shape[1] * kernel_channel_size;
    let in_channel_size = in_shape[2] * in_shape[3];
    let in_size = in_shape[1] * in_channel_size;

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let weight_data = if let DTypes::F32(data) = weight.get_data() { data } else { panic!("Weight data type does not match input data type!") };
            let bias_data = bias.map_or(None, |tensor| if let DTypes::F32(data) = tensor.get_data() { Some(data.clone()) } else { panic!("Bias data type does not match input data type!!") });

            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x) {
                            let mut sum = 0.;
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;

                                    if cy >= 0 && cy < in_shape[2] as isize && cx >= 0 && cx < in_shape[3] as isize {
                                        for kc in 0..weight_shape[1] {
                                            let ker_idx = c * kernel_size + kc * kernel_channel_size + ky * weight_shape[3] + kx;
                                            let cur_idx = n * in_size + c * in_channel_size + cy as usize * in_shape[3] + cx as usize;
                                            
                                            let ker_val = weight_data[ker_idx];
                                            let cur_val = arr[cur_idx];

                                            sum += ker_val * cur_val;
                                        }
                                    }
                                }
                            }
                            out_data[out_idx] = sum + bias_data.as_ref().map_or(0., |bias| bias[c]);
                            out_idx += 1;
                        }
                    }
                }
            }

            DTypes::F32(out_data)
        },
        DTypes::F64(arr) => {
            let weight_data = if let DTypes::F64(data) = weight.get_data() { data } else { panic!("Weight data type does not match input data type!") };
            let bias_data = bias.map_or(None, |tensor| if let DTypes::F64(data) = tensor.get_data() { Some(data.clone()) } else { panic!("Bias data type does not match input data type!!") });

            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x) {
                            let mut sum = 0.;
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;

                                    if cy >= 0 && cy < in_shape[2] as isize && cx >= 0 && cx < in_shape[3] as isize {
                                        for kc in 0..weight_shape[1] {
                                            let ker_idx = c * kernel_size + kc * kernel_channel_size + ky * weight_shape[3] + kx;
                                            let cur_idx = n * in_size + c * in_channel_size + cy as usize * in_shape[3] + cx as usize;
                                            
                                            let ker_val = weight_data[ker_idx];
                                            let cur_val = arr[cur_idx];

                                            sum += ker_val * cur_val;
                                        }
                                    }
                                }
                            }
                            out_data[out_idx] = sum + bias_data.as_ref().map_or(0., |bias| bias[c]);
                            out_idx += 1;
                        }
                    }
                }
            }

            DTypes::F64(out_data)
        },
        _ => { panic!("Convolutional does not support these data type!!") }
    };

    output.set_data(out_data);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_conv() {
        let input = Tensor::new(DTypes::F64(vec![            
            0., 1., 2., 3., 4., 
            5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14.,
            15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.
        ]), vec![1, 1, 5, 5]);
        let weight = Tensor::new(DTypes::F64(vec![
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.
        ]), vec![1, 1, 3, 3]);

        let output = handle_conv(3, 3, 0, 0, 0, 0, 1, 1, &input, &weight, None);
        assert_eq!(output.get_shape(), vec![1, 1, 3, 3]);
        assert_eq!(*output.get_data(), DTypes::F64(vec![54., 63., 72., 99., 108., 117., 144., 153., 162.]));
    }

    #[test]
    fn test_conv_with_padding() {
        let input = Tensor::new(DTypes::F64(vec![            
            0., 1., 2., 3., 4., 
            5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14.,
            15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.
        ]), vec![1, 1, 5, 5]);
        let weight = Tensor::new(DTypes::F64(vec![
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.
        ]), vec![1, 1, 3, 3]);

        let output = handle_conv(3, 3, 1, 1, 1, 1, 1, 1, &input, &weight, None);
        assert_eq!(output.get_shape(), vec![1, 1, 5, 5]);
        assert_eq!(*output.get_data(), DTypes::F64(vec![
            12., 21., 27., 33., 24., 33., 54., 63., 72., 
            51., 63., 99., 108., 117., 81., 93., 144.,
            153., 162., 111., 72., 111., 117., 123., 84.,
        ]));
    }

    #[test]
    fn test_conv_with_stride() {
        let input = Tensor::new(DTypes::F64(vec![            
            0., 1., 2., 3., 4., 
            5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14.,
            15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
            25., 26., 27., 28., 29.,
            30., 31., 32., 33., 34.,
        ]), vec![1, 1, 7, 5]);
        let weight = Tensor::new(DTypes::F64(vec![
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.
        ]), vec![1, 1, 3, 3]);

        let output = handle_conv(3, 3, 0, 0, 0, 0, 2, 2, &input, &weight, None);
        assert_eq!(output.get_shape(), vec![1, 1, 3, 2]);
        assert_eq!(*output.get_data(), DTypes::F64(vec![
            54., 72., 144., 162., 234., 252.
        ]));
    }

    #[test]
    fn test_conv_with_stride_and_paddings() {
        let input = Tensor::new(DTypes::F64(vec![            
            0., 1., 2., 3., 4., 
            5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14.,
            15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
            25., 26., 27., 28., 29.,
            30., 31., 32., 33., 34.,
        ]), vec![1, 1, 7, 5]);
        let weight = Tensor::new(DTypes::F64(vec![
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.
        ]), vec![1, 1, 3, 3]);

        let output = handle_conv(3, 3, 1, 0, 1, 0, 2, 2, &input, &weight, None);
        assert_eq!(output.get_shape(), vec![1, 1, 4, 2]);
        assert_eq!(*output.get_data(), DTypes::F64(vec![
            21., 33., 99., 117., 189., 207., 171., 183.
        ]));
    }
}

