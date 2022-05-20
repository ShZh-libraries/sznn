use wasm_bindgen::prelude::*;

use crate::Tensor;

#[wasm_bindgen(js_name = handleDropout)]
pub fn handle_dropout(input: &Tensor) -> Tensor {
    input.clone()
}
