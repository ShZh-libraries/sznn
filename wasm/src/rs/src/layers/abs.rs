use wasm_bindgen::prelude::*;

use crate::tensor::{TensorDataType, Tensor};


#[wasm_bindgen]
pub fn forward(input: &Tensor) -> Tensor {
    let mut output = Tensor::new();
    output.set_vec_shape(input.get_shape());
    output.data = match &input.data {
        TensorDataType::Int8(arr) => {
            let out = arr
                .iter()
                .map(|x| if *x < 0 { -*x } else { *x } )
                .collect::<Vec<_>>();
            TensorDataType::Int8(out)
        },
        TensorDataType::Int16(arr) => {
            let out = arr
                .iter()
                .map(|x| if *x < 0 { -*x } else { *x } )
                .collect::<Vec<_>>();
            TensorDataType::Int16(out)
        }
        TensorDataType::Int32(arr) => {
            let out = arr
                .iter()
                .map(|x| if *x < 0 { -*x } else { *x } )
                .collect::<Vec<_>>();
            TensorDataType::Int32(out)
        }
        TensorDataType::Float32(arr) => {
            let out = arr
                .iter()
                .map(|x| if *x < 0. { -*x } else { *x } )
                .collect::<Vec<_>>();
            TensorDataType::Float32(out)
        }
        TensorDataType::Float64(arr) => {
            let out = arr
                .iter()
                .map(|x| if *x < 0. { -*x } else { *x } )
                .collect::<Vec<_>>();
            TensorDataType::Float64(out)
        }
        _ => {
            panic!("Operation not supported on target type!");
        }
    };

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_forward() {
        let mut input = Tensor::new();
        input.set_vec_shape(vec![2, 2]);
        input.data = TensorDataType::Int32(vec![1, -2, -3, 4]);
    
        let output = forward(&input);
    
        assert_eq!(output.get_length(), 4);
        assert_eq!(output.get_ndim(), 2);
        assert_eq!(output.get_shape(), vec![2, 2]);
        assert_eq!(output.data, TensorDataType::Int32(vec![1, 2, 3, 4]));       
    }
}

