use rayon::prelude::*;
use wasm_bindgen::prelude::*;

use crate::tensor::{DTypes, Tensor};

#[wasm_bindgen(js_name = handleSign)]
pub fn handle_sign(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => {
            let out = arr.par_iter().map(|x| x.signum()).collect::<Vec<_>>();
            DTypes::I8(out)
        }
        DTypes::I16(arr) => {
            let out = arr.par_iter().map(|x| x.signum()).collect::<Vec<_>>();
            DTypes::I16(out)
        }
        DTypes::I32(arr) => {
            let out = arr.par_iter().map(|x| x.signum()).collect::<Vec<_>>();
            DTypes::I32(out)
        }
        DTypes::F32(arr) => {
            let out = arr.par_iter().map(|x| x.signum()).collect::<Vec<_>>();
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let out = arr.par_iter().map(|x| x.signum()).collect::<Vec<_>>();
            DTypes::F64(out)
        }
        _ => {
            panic!("Operation not supported on target type!");
        }
    };

    Tensor::new(out_data, out_shape)
}
