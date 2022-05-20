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
    for (i, item) in out_shape.iter_mut().enumerate().take(len) {
        if !dims.includes(&JsValue::from_f64(i as f64), 0) {
            *item = in_shape[in_idx];
            in_idx += 1;
        }
    }

    Tensor::new(out_data, out_shape)
}
