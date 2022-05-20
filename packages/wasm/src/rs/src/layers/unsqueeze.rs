use js_sys::Array;
use wasm_bindgen::prelude::*;

use crate::Tensor;

#[wasm_bindgen(js_name = handleUnsqueeze)]
pub fn handle_unsqueeze(input: &Tensor, dims: &Array) -> Tensor {
    let out_data = input.get_data().clone();

    let in_shape = input.get_shape();
    let len = input.get_dim() + dims.length() as usize;
    let mut out_shape = vec![1; len];
    let mut in_idx = 0;
    for i in 0..len {
        if !dims.includes(&JsValue::from_f64(i as f64), 0) {
            out_shape[i] = in_shape[in_idx];
            in_idx += 1;
        }
    }

    Tensor::new(out_data, out_shape)
}
