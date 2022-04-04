use crate::{Tensor, DTypes};

pub fn forwardMaxPool2D(
    input: &Tensor,
    kernel_height: usize, kernel_width: usize,
    pad_top: usize, pad_left: usize, 
    pad_bottom: usize, pad_right: usize, 
    stride_y: usize, stride_x: usize,
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
                        for x in (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x) {
                            let max_idx = n as isize * in_size as isize + c as isize * in_channel_size as isize + y * in_shape[3] as isize + x;
                            let mut max_val = arr[max_idx as usize];
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0 && cy < in_shape[2] as isize && cx >= 0 && cx < in_shape[3] as isize {
                                        let cur_idx = n * in_size + c * in_channel_size + cy as usize * in_shape[3] + cx as usize;
                                        arr[cur_idx]
                                    } else { 0. };
        
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
        },
        DTypes::F64(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize)..=(max_y - pad_top) as isize).step_by(stride_y) {
                        for x in (-(pad_left as isize)..=(max_x - pad_left) as isize).step_by(stride_x) {
                            let max_idx = n as isize * in_size as isize + c as isize * in_channel_size as isize + y * in_shape[3] as isize + x;
                            let mut max_val = arr[max_idx as usize];
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let cy = y + ky as isize;
                                    let cx = x + kx as isize;
                                    let val = if cy >= 0 && cy < in_shape[2] as isize && cx >= 0 && cx < in_shape[3] as isize {
                                        let cur_idx = n * in_size + c * in_channel_size + cy as usize * in_shape[3] + cx as usize;
                                        arr[cur_idx]
                                    } else { 0. };
        
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
        },
        _ => panic!("Max pooling does not support these data types.")
    };
    output.set_data(out_data);
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pooling() {
        let input = Tensor::new(
            DTypes::F32(vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9.,
                10., 11., 12., 13., 14., 15., 16., 17., 18., 19.
            ]),
            vec![1, 2, 3, 3]
        );
    
        let output = forwardMaxPool2D(&input, 2, 2, 0, 0, 0, 0, 1, 1);
    
        assert_eq!(output.get_shape(), vec![1, 2, 2, 2]);
        assert_eq!(*output.get_data(), DTypes::F32(vec![5., 6., 8., 9., 14., 15., 17., 18.]));
    }
}

