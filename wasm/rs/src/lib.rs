use wasm_bindgen::prelude::*;
pub mod tensor;

#[wasm_bindgen]
pub fn add(a: u32, b: u32) -> u32 {
    a + b
}