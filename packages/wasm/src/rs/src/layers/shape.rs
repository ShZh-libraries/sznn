use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor};

macro_rules! cast {
    ($shape: expr, $typ: ty) => {{
        $shape.iter().map(|&x| x as $typ).collect()
    }};
}

#[wasm_bindgen(js_name = handleShape)]
pub fn handle_shape(input: &Tensor) -> Tensor {
    let shape = input.get_shape();
    let out_shape = vec![shape.len()];
    let out_data = match &input.get_data() {
        DTypes::I8(_) => DTypes::I8(cast!(shape, i8)),
        DTypes::I16(_) => DTypes::I16(cast!(shape, i16)),
        DTypes::I32(_) => DTypes::I32(cast!(shape, i32)),
        DTypes::U8(_) => DTypes::U8(cast!(shape, u8)),
        DTypes::U16(_) => DTypes::U16(cast!(shape, u16)),
        DTypes::U32(_) => DTypes::U32(cast!(shape, u32)),
        DTypes::F32(_) => DTypes::F32(cast!(shape, f32)),
        DTypes::F64(_) => DTypes::F64(cast!(shape, f64)),
    };

    Tensor::new(out_data, out_shape)
}
