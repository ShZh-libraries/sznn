use crate::tensor::Tensor;

pub fn forward(input: &Tensor, shape: &mut Vec<isize>) -> Tensor {
    // Deal with zero
    let input_shape = input.get_shape();
    for index in 0..shape.len() {
        if shape[index] == 0 {
            shape[index] = input_shape[index] as isize;
        }
    }

    // Deal with negative numbers
    let negative_idx = shape.iter().position(|&x| x == -1);
    if let Some(idx) = negative_idx {
        let remain_len = shape.iter().fold(1, |res, val| res * val) as usize;
        let remain_size = input.get_length() / remain_len;
        shape[idx] = remain_size as isize;
    }

    let out_data = input.get_data().clone();
    let out_shape = shape.iter().map(|&x| x as usize).collect();

    Tensor::new(out_data, out_shape)
}