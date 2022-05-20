use wasm_bindgen::prelude::wasm_bindgen;

use crate::{tensor::Tensor, DTypes};

use super::extract_data;

#[wasm_bindgen(js_name = handleGather)]
pub fn handle_gather(input: &Tensor, indices: &Tensor) -> Tensor {
    let out_shape = vec![1];
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => {
            let indices = extract_data!(indices, DTypes::I8);
            DTypes::I8(vec![arr[indices[0] as usize]])
        }
        DTypes::I16(arr) => {
            let indices = extract_data!(indices, DTypes::I16);
            DTypes::I16(vec![arr[indices[0] as usize]])
        }
        DTypes::I32(arr) => {
            let indices = extract_data!(indices, DTypes::I32);
            DTypes::I32(vec![arr[indices[0] as usize]])
        }
        DTypes::U8(arr) => {
            let indices = extract_data!(indices, DTypes::U8);
            DTypes::U8(vec![arr[indices[0] as usize]])
        }
        DTypes::U16(arr) => {
            let indices = extract_data!(indices, DTypes::U16);
            DTypes::U16(vec![arr[indices[0] as usize]])
        }
        DTypes::U32(arr) => {
            let indices = extract_data!(indices, DTypes::U32);
            DTypes::U32(vec![arr[indices[0] as usize]])
        }
        DTypes::F32(arr) => {
            let indices = extract_data!(indices, DTypes::F32);
            DTypes::F32(vec![arr[indices[0] as usize]])
        }
        DTypes::F64(arr) => {
            let indices = extract_data!(indices, DTypes::F64);
            DTypes::F64(vec![arr[indices[0] as usize]])
        }
    };

    Tensor::new(out_data, out_shape)
}
