use wasm_bindgen::prelude::*;

use crate::tensor::{DTypes, Tensor};

#[wasm_bindgen(js_name = handleAbs)]
pub fn handle_abs(input: &Tensor) -> Tensor {
    let out_shape = input.get_shape();
    let out_data = match &input.get_data() {
        DTypes::I8(arr) => {
            let out = arr.iter().map(|x| x.abs()).collect::<Vec<_>>();
            DTypes::I8(out)
        }
        DTypes::I16(arr) => {
            let out = arr.iter().map(|x| x.abs()).collect::<Vec<_>>();
            DTypes::I16(out)
        }
        DTypes::I32(arr) => {
            let out = arr.iter().map(|x| x.abs()).collect::<Vec<_>>();
            DTypes::I32(out)
        }
        DTypes::F32(arr) => {
            let out = arr.iter().map(|x| x.abs()).collect::<Vec<_>>();
            DTypes::F32(out)
        }
        DTypes::F64(arr) => {
            let out = arr.iter().map(|x| x.abs()).collect::<Vec<_>>();
            DTypes::F64(out)
        }
        _ => {
            panic!("Operation not supported on target type!");
        }
    };

    Tensor::new(out_data, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_forward() {
        let input = Tensor::new(DTypes::I32(vec![1, -2, -3, 4]), vec![2, 2]);

        let output = handle_abs(&input);

        assert_eq!(output.get_length(), 4);
        assert_eq!(output.get_dim(), 2);
        assert_eq!(output.get_shape(), vec![2, 2]);
        assert_eq!(*output.get_data(), DTypes::I32(vec![1, 2, 3, 4]));
    }
}
