use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::tensor::{DTypes, Tensor};

#[wasm_bindgen(js_name = handleTan)]
pub fn handle_tan(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let out = arr.par_iter().map(|x| x.tan()).collect::<Vec<_>>();
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let out = arr.par_iter().map(|x| x.tan()).collect::<Vec<_>>();
            DTypes::F64(out)
        }
        _ => {
            panic!("Operation not supported on target type!");
        }
    };

    Tensor::new(out_data, out_shape)
}
