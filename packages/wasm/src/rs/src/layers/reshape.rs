use wasm_bindgen::prelude::wasm_bindgen;

use crate::{tensor::Tensor, DTypes};

#[wasm_bindgen(js_name = handleReshape)]
pub fn handle_reshape(input: &Tensor, shape: Tensor) -> Tensor {
    let mut shape = if let DTypes::I32(arr) = shape.get_data() {
        arr.clone()
    } else {
        panic!("The shape's dtype is not i32!!")
    };

    // Deal with zero
    let input_shape = input.get_shape();
    for index in 0..shape.len() {
        if shape[index] == 0 {
            shape[index] = input_shape[index] as i32;
        }
    }

    // Deal with negative numbers
    let negative_idx = shape.iter().position(|&x| x == -1);
    if let Some(idx) = negative_idx {
        let remain_len = shape.iter().filter(|x| x.is_positive()).product::<i32>() as usize;
        let remain_size = input.get_length() / remain_len;
        shape[idx] = remain_size as i32;
    }

    let out_data = input.get_data().clone();
    let out_shape = shape.iter().map(|&x| x as usize).collect();

    Tensor::new(out_data, out_shape)
}
