use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor};

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
    let out_height = in_shape[2] + pad_top + pad_bottom;
    let out_width = in_shape[3] + pad_left + pad_right;
    let out_shape = vec![in_shape[0], in_shape[1], out_height, out_width];
    output.set_shape(out_shape);

    let in_channel_size = in_shape[2] * in_shape[3];
    let in_size = in_shape[1] * in_channel_size;

    let out_data = match &input.get_data() {
        DTypes::F32(arr) => {
            let mut out_idx = 0;
            let mut out_data = vec![0.; output.get_length()];
            for n in 0..in_shape[0] {
                for c in 0..in_shape[1] {
                    for y in (-(pad_top as isize))..(out_height - pad_top) as isize {
                        for x in (-(pad_left as isize))..(out_width - pad_left) as isize {
                            if y < 0 || y >= in_shape[2] as isize || x < 0 || x >= in_shape[3] as isize {
                                out_data[out_idx] = 0.;
                            } else {
                                let in_idx = n * in_size + c * in_channel_size + y as usize * in_shape[3] + x as usize;
                                out_data[out_idx] = arr[in_idx];
                            }

                            out_idx += 1;
                        }
                    }
                }
            }

            DTypes::F32(out_data)
        }
        _ => {
            panic!("Data type not support!");
        }
    };
    output.set_data(out_data);

    output
}