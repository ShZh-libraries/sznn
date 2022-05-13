#![allow(non_snake_case)]

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{DTypes, Tensor, TensorList};

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
    let transformed_tensors = inputs
        .iter()
        .map(|input| {
            let height;
            let width;
            let shape = input.get_shape();
            if axis != 0 {
                height = shape[..axis].iter().fold(1, |sum, val| sum * *val);
                width = shape[axis..].iter().fold(1, |sum, val| sum * *val);
            } else {
                height = 1;
                width = shape.iter().fold(1, |sum, val| sum * *val);
            }

            input.clone().reshape(vec![height, width])
        })
        .collect();

    let mut transformed_result = concat_2D_axis_1(transformed_tensors);
    transformed_result.reshape(out_shape)
}

fn get_concat_shape(inputs: &Vec<Tensor>, axis: usize) -> Vec<usize> {
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
        DTypes::I8(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I8(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::I8(out_data)
        }
        DTypes::I16(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I16(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::I16(out_data)
        }
        DTypes::I32(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I32(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::I32(out_data)
        }
        DTypes::U8(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U8(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::U8(out_data)
        }
        DTypes::U16(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U16(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::U16(out_data)
        }
        DTypes::U32(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U32(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::U32(out_data)
        }
        DTypes::F32(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::F32(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::F32(out_data)
        }
        DTypes::F64(_) => {
            let mut out_data = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::F64(data) = &*(*input).get_data() {
                    data
                } else {
                    panic!("The input data's type does not match")
                };
                let shape = input.get_shape();

                for idx in 0..shape[0] {
                    out_data.push(data[idx]);
                }
            }

            DTypes::F64(out_data)
        }
    };

    Tensor::new(out_data, vec![out_shape])
}

fn concat_2D_axis_0(inputs: Vec<Tensor>) -> Tensor {
    let out_shape = get_concat_shape(&inputs, 0);

    let shape = inputs[0].get_shape();
    let (height, width) = (shape[0], shape[1]);

    let out_data = match *inputs[0].get_data() {
        DTypes::I8(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I8(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I8(out)
        }
        DTypes::I16(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I16(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I16(out)
        }
        DTypes::I32(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::I32(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I32(out)
        }
        DTypes::U8(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U8(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U8(out)
        }
        DTypes::U16(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U16(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U16(out)
        }
        DTypes::U32(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::U32(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U32(out)
        }
        DTypes::F32(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::F32(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::F32(out)
        }
        DTypes::F64(_) => {
            let mut out = Vec::new();
            for input in inputs.iter() {
                let data = if let DTypes::F64(arr) = input.get_data() {
                    &*arr
                } else {
                    panic!("The input data's type does not match")
                };

                for y in 0..height {
                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::F64(out)
        }
    };

    Tensor::new(out_data, out_shape)
}

fn concat_2D_axis_1(inputs: Vec<Tensor>) -> Tensor {
    let out_shape = get_concat_shape(&inputs, 0);

    let shape = inputs[0].get_shape();
    let (height, width) = (shape[0], shape[1]);

    let out_data = match *inputs[0].get_data() {
        DTypes::I8(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::I8(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I8(out)
        }
        DTypes::I16(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::I16(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I16(out)
        }
        DTypes::I32(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::I32(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::I32(out)
        }
        DTypes::U8(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::U8(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U8(out)
        }
        DTypes::U16(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::U16(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U16(out)
        }
        DTypes::U32(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::U32(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::U32(out)
        }
        DTypes::F32(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::F32(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::F32(out)
        }
        DTypes::F64(_) => {
            let mut out = Vec::new();

            for y in 0..height {
                for input in inputs.iter() {
                    let data = if let DTypes::F64(arr) = input.get_data() {
                        &*arr
                    } else {
                        panic!("The input data's type does not match")
                    };

                    for x in 0..width {
                        out.push(data[y * width + x]);
                    }
                }
            }

            DTypes::F64(out)
        }
    };

    Tensor::new(out_data, out_shape)
}
