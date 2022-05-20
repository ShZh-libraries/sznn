use js_sys::Array;
use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};

#[wasm_bindgen(js_name = handleSlice)]
pub fn handle_slice(input: &Tensor, axes: &Array, starts: &Array, ends: &Array) -> Tensor {
    if axes.length() == 1 && axes.get(0).as_f64().unwrap() == 0. {
        let start = starts.get(0).as_f64().unwrap() as usize;
        let end = ends.get(0).as_f64().unwrap() as usize;
        let out_shape = vec![end - start];

        let out_data = match &input.get_data() {
            DTypes::I8(arr) => DTypes::I8(arr[start..end].to_vec()),
            DTypes::I16(arr) => DTypes::I16(arr[start..end].to_vec()),
            DTypes::I32(arr) => DTypes::I32(arr[start..end].to_vec()),
            DTypes::U8(arr) => DTypes::U8(arr[start..end].to_vec()),
            DTypes::U16(arr) => DTypes::U16(arr[start..end].to_vec()),
            DTypes::U32(arr) => DTypes::U32(arr[start..end].to_vec()),
            DTypes::F32(arr) => DTypes::F32(arr[start..end].to_vec()),
            DTypes::F64(arr) => DTypes::F64(arr[start..end].to_vec()),
        };

        return Tensor::new(out_data, out_shape);
    }

    Tensor::new_empty()
}
