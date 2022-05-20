use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor};

macro_rules! padding {
    (
        $arr: expr, $in_shape: expr, $out_shape: expr, $len: expr,
        $pad_top: expr, $pad_left: expr, $pad_bottom: expr, $pad_right: expr
    ) => {{
        let in_channel_size = $in_shape[2] * $in_shape[3];
        let in_size = $in_shape[1] * in_channel_size;

        let mut out_idx = 0;
        let mut out_data = vec![0.; $len];
        for n in 0..$out_shape[0] {
            for c in 0..$out_shape[1] {
                for y in (-($pad_top as isize))..($out_shape[2] - $pad_top) as isize {
                    for x in (-($pad_left as isize))..($out_shape[3] - $pad_left) as isize {
                        if y < 0
                            || y >= $in_shape[2] as isize
                            || x < 0
                            || x >= $in_shape[3] as isize
                        {
                            out_data[out_idx] = 0.;
                        } else {
                            let in_idx = n * in_size
                                + c * in_channel_size
                                + y as usize * $in_shape[3]
                                + x as usize;
                            out_data[out_idx] = $arr[in_idx];
                        }

                        out_idx += 1;
                    }
                }
            }
        }

        out_data
    }};
}

#[wasm_bindgen(js_name = handlePadding)]
pub fn handle_padding(
    input: &Tensor,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Tensor {
    let mut output = Tensor::new_empty();

    let in_shape = input.get_shape();
    let out_shape = vec![
        in_shape[0],
        in_shape[1],
        in_shape[2] + pad_top + pad_bottom,
        in_shape[3] + pad_left + pad_right,
    ];
    output.set_shape(out_shape.clone());

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => DTypes::F32(padding!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            pad_top,
            pad_left,
            pad_bottom,
            pad_right
        )),
        DTypes::F64(arr) => DTypes::F64(padding!(
            arr,
            in_shape,
            out_shape,
            output.get_length(),
            pad_top,
            pad_left,
            pad_bottom,
            pad_right
        )),
        _ => panic!("Current type is not support in padding layer!"),
    };
    output.set_data(out_data);

    output
}
