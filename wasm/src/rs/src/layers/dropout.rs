use wasm_bindgen::prelude::*;

use crate::Tensor;

#[wasm_bindgen]
pub fn forward(input: &Tensor) -> Tensor {
    input.clone()
}