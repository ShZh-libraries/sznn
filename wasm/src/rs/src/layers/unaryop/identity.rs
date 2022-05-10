use wasm_bindgen::prelude::*;

use crate::tensor::Tensor;

#[wasm_bindgen(js_name = handleIdentity)]
pub fn handle_identity(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = input.get_data();

    Tensor::new(out_data.clone(), out_shape)
}
