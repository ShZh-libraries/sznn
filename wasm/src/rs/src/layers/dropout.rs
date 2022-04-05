use wasm_bindgen::prelude::*;

use crate::Tensor;

#[wasm_bindgen]
pub fn handle_dropout(input: &Tensor) -> Tensor {
    input.clone()
}