use wasm_bindgen::prelude::*;

use crate::{DTypes, Tensor};

use super::extract_data;

macro_rules! upsample {
    ($arr: expr, $scale: expr, $in_shape: expr, $len: expr, $typ: ty) => {{
        let channel_size = $in_shape[2] * $in_shape[3];
        let size = $in_shape[1] * channel_size;

        let mut out_idx = 0;
        let mut out = vec![0 as $typ; $len];
        for n in 0..$in_shape[0] {
            for _ in 0..$scale[0] as usize {
                for c in 0..$in_shape[1] {
                    for _ in 0..$scale[1] as usize {
                        for y in 0..$in_shape[2] {
                            for _ in 0..$scale[2] as usize {
                                for x in 0..$in_shape[3] {
                                    for _ in 0..$scale[3] as usize {
                                        let idx =
                                            n * size + c * channel_size + y * $in_shape[3] + x;
                                        out[out_idx] = $arr[idx];
                                        out_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }};
}

#[wasm_bindgen(js_name = handleUpSample)]
pub fn handle_upsample(input: &Tensor, scale: &Tensor) -> Tensor {
    let mut output = Tensor::new_empty();
    let scale = extract_data!(scale, DTypes::F32);
    let in_shape = input.get_shape();
    let out_shape = in_shape
        .iter()
        .zip(scale.iter())
        .map(|(dim, scale)| (*scale * *dim as f32) as usize)
        .collect::<Vec<_>>();
    output.set_shape(out_shape.clone());
    let len = output.get_length();

    let out_data = match &input.get_data() {
        DTypes::I8(arr) => DTypes::I8(upsample!(arr, scale, in_shape, len, i8)),
        DTypes::I16(arr) => DTypes::I16(upsample!(arr, scale, in_shape, len, i16)),
        DTypes::I32(arr) => DTypes::I32(upsample!(arr, scale, in_shape, len, i32)),
        DTypes::U8(arr) => DTypes::U8(upsample!(arr, scale, in_shape, len, u8)),
        DTypes::U16(arr) => DTypes::U16(upsample!(arr, scale, in_shape, len, u16)),
        DTypes::U32(arr) => DTypes::U32(upsample!(arr, scale, in_shape, len, u32)),
        DTypes::F32(arr) => DTypes::F32(upsample!(arr, scale, in_shape, len, f32)),
        DTypes::F64(arr) => DTypes::F64(upsample!(arr, scale, in_shape, len, f64)),
    };

    Tensor::new(out_data, out_shape)
}
