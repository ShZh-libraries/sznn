use crate::{Tensor, TensorDataType};

pub fn forward(input: &Tensor) -> Tensor {
    let mut output = Tensor::new();

    let shape = input.get_shape();
    output.set_vec_shape(vec![shape.len()]);
    output.data = match &input.data {
        TensorDataType::Int8(_) => {
            let out = shape.iter().map(|&x| x as i8).collect();
            TensorDataType::Int8(out)
        },
        TensorDataType::Int16(_) => {
            let out = shape.iter().map(|&x| x as i16).collect();
            TensorDataType::Int16(out)
        },
        TensorDataType::Int32(_) => {
            let out = shape.iter().map(|&x| x as i32).collect();
            TensorDataType::Int32(out)
        },
        TensorDataType::UInt8(_) => {
            let out = shape.iter().map(|&x| x as u8).collect();
            TensorDataType::UInt8(out)
        },
        TensorDataType::UInt16(_) => {
            let out = shape.iter().map(|&x| x as u16).collect();
            TensorDataType::UInt16(out)
        },
        TensorDataType::UInt32(_) => {
            let out = shape.iter().map(|&x| x as u32).collect();
            TensorDataType::UInt32(out)
        },
        TensorDataType::Float32(_) => {
            let out = shape.iter().map(|&x| x as f32).collect();
            TensorDataType::Float32(out)
        },
        TensorDataType::Float64(_) => {
            let out = shape.iter().map(|&x| x as f64).collect();
            TensorDataType::Float64(out)
        },
    };

    output
}
