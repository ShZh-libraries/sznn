#![allow(non_snake_case)]

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor, TensorList};

use super::extract_data;

macro_rules! concat_1d {
    ($inputs: expr, $typ: path) => {{
        let mut inputs_data = Vec::new();
        for input in $inputs.iter() {
            let data = extract_data!(input, $typ);
            inputs_data.push(data);
        }

        inputs_data
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
    }};
}

macro_rules! concat_2d_axis_0 {
    ($inputs: expr, $typ: path) => {{
        concat_1d!($inputs, $typ)
    }};
}

macro_rules! concat_2d_axis_1 {
    ($inputs: expr, $typ: path, $height: expr, $width: expr) => {{
        let mut inputs_data = Vec::new();
        let mut offset = 0;

        (0..$height).into_iter().for_each(|_| {
            for input in $inputs.iter() {
                let data = extract_data!(input, $typ);

                inputs_data.push(&data[offset..(offset + $width)]);
            }

            offset += $width;
        });

        inputs_data
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
    }};
}

#[wasm_bindgen(js_name = handleConcat)]
pub fn handle_concat(inputs: TensorList, axis: usize) -> Tensor {
    let inputs = inputs.get();

    if inputs.len() == 1 {
        return inputs[0].clone();
    }
    if inputs[0].get_dim() == 1 {
        return concat_1D(inputs);
    }
    if inputs[0].get_dim() == 2 {
        if axis == 0 {
            return concat_2D_axis_0(inputs);
        } else if axis == 1 {
            return concat_2D_axis_1(inputs);
        } else {
            panic!("Axis exceed input's dim!");
        }
    }

    let out_shape = get_concat_shape(&inputs, axis);
    let transformed = inputs
        .iter()
        .map(|input| {
            let shape = input.get_shape();
            let (height, width) = if axis != 0 {
                (
                    shape[..axis].iter().product(),
                    shape[axis..].iter().product(),
                )
            } else {
                (1, shape.iter().product())
            };

            input.clone().reshape(vec![height, width])
        })
        .collect();

    let mut result = concat_2D_axis_1(transformed);
    result.reshape(out_shape)
}

fn get_concat_shape(inputs: &[Tensor], axis: usize) -> Vec<usize> {
    let mut axis_size = 0;
    for data in inputs.iter() {
        axis_size += data.get_shape()[axis];
    }

    let mut result = inputs[0].get_shape();
    result[axis] = axis_size;

    result
}

fn concat_1D(inputs: Vec<Tensor>) -> Tensor {
    let out_shape = inputs
        .iter()
        .map(|x| x.get_shape()[0])
        .reduce(|res, val| res + val)
        .unwrap();

    let out_data = match *inputs[0].get_data() {
        DTypes::I8(_) => DTypes::I8(concat_1d!(inputs, DTypes::I8)),
        DTypes::I16(_) => DTypes::I16(concat_1d!(inputs, DTypes::I16)),
        DTypes::I32(_) => DTypes::I32(concat_1d!(inputs, DTypes::I32)),
        DTypes::U8(_) => DTypes::U8(concat_1d!(inputs, DTypes::U8)),
        DTypes::U16(_) => DTypes::U16(concat_1d!(inputs, DTypes::U16)),
        DTypes::U32(_) => DTypes::U32(concat_1d!(inputs, DTypes::U32)),
        DTypes::F32(_) => DTypes::F32(concat_1d!(inputs, DTypes::F32)),
        DTypes::F64(_) => DTypes::F64(concat_1d!(inputs, DTypes::F64)),
    };

    Tensor::new(out_data, vec![out_shape])
}

fn concat_2D_axis_0(inputs: Vec<Tensor>) -> Tensor {
    let out_shape = get_concat_shape(&inputs, 0);

    let out_data = match *inputs[0].get_data() {
        DTypes::I8(_) => DTypes::I8(concat_2d_axis_0!(inputs, DTypes::I8)),
        DTypes::I16(_) => DTypes::I16(concat_2d_axis_0!(inputs, DTypes::I16)),
        DTypes::I32(_) => DTypes::I32(concat_2d_axis_0!(inputs, DTypes::I32)),
        DTypes::U8(_) => DTypes::U8(concat_2d_axis_0!(inputs, DTypes::U8)),
        DTypes::U16(_) => DTypes::U16(concat_2d_axis_0!(inputs, DTypes::U16)),
        DTypes::U32(_) => DTypes::U32(concat_2d_axis_0!(inputs, DTypes::U32)),
        DTypes::F32(_) => DTypes::F32(concat_2d_axis_0!(inputs, DTypes::F32)),
        DTypes::F64(_) => DTypes::F64(concat_2d_axis_0!(inputs, DTypes::F64)),
    };

    Tensor::new(out_data, out_shape)
}

fn concat_2D_axis_1(inputs: Vec<Tensor>) -> Tensor {
    let out_shape = get_concat_shape(&inputs, 0);

    let shape = inputs[0].get_shape();
    let (height, width) = (shape[0], shape[1]);

    let out_data = match *inputs[0].get_data() {
        DTypes::I8(_) => DTypes::I8(concat_2d_axis_1!(inputs, DTypes::I8, height, width)),
        DTypes::I16(_) => DTypes::I16(concat_2d_axis_1!(inputs, DTypes::I16, height, width)),
        DTypes::I32(_) => DTypes::I32(concat_2d_axis_1!(inputs, DTypes::I32, height, width)),
        DTypes::U8(_) => DTypes::U8(concat_2d_axis_1!(inputs, DTypes::U8, height, width)),
        DTypes::U16(_) => DTypes::U16(concat_2d_axis_1!(inputs, DTypes::U16, height, width)),
        DTypes::U32(_) => DTypes::U32(concat_2d_axis_1!(inputs, DTypes::U32, height, width)),
        DTypes::F32(_) => DTypes::F32(concat_2d_axis_1!(inputs, DTypes::F32, height, width)),
        DTypes::F64(_) => DTypes::F64(concat_2d_axis_1!(inputs, DTypes::F64, height, width)),
    };

    Tensor::new(out_data, out_shape)
}
