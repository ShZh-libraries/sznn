use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub enum DType {
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
    Float32,
    Float64,
}

#[derive(PartialEq, Debug, Clone)]
pub enum DTypes {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Tensor {
    data: DTypes,
    shape: Vec<usize>,
}

#[wasm_bindgen]
pub struct TensorList {
    tensors: Vec<Tensor>
}

#[wasm_bindgen]
impl Tensor {
    #[wasm_bindgen(constructor)]
    pub fn new_empty() -> Self {
        console_error_panic_hook::set_once();
        Self {
            data: DTypes::F32(Vec::new()),
            shape: Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape_wiht_array(&mut self, shape: &js_sys::Array) {
        let mut tensor_shape = Vec::with_capacity(shape.length() as usize);
        shape.iter().for_each(|num| tensor_shape.push(num.as_f64().unwrap() as usize));

        self.shape = tensor_shape;
    }

    #[wasm_bindgen(js_name = setDataWithArray)]
    pub fn set_data_with_array(&mut self, data: &js_sys::Array, dtype: DType) {
        match dtype {
            DType::Int8 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i8;
                });
                self.data = DTypes::I8(tensor_data);
            }
            DType::Int16 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i16;
                });
                self.data = DTypes::I16(tensor_data);
            }
            DType::Int32 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i32;
                });
                self.data = DTypes::I32(tensor_data);
            }
            DType::UInt8 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u8;
                });
                self.data = DTypes::U8(tensor_data);
            }
            DType::UInt16 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u16;
                });
                self.data = DTypes::U16(tensor_data);
            }
            DType::UInt32 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u32;
                });
                self.data = DTypes::U32(tensor_data);
            }
            DType::Float32 => {
                let mut tensor_data = vec![0.; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as f32;
                });
                self.data = DTypes::F32(tensor_data);
            }
            DType::Float64 => {
                let mut tensor_data = vec![0.; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap();
                });
                self.data = DTypes::F64(tensor_data);
            }
        }
    }

    #[wasm_bindgen(js_name = setDataWithI8Array)]
    pub fn set_data_with_i8_array(&mut self, data: &js_sys::Int8Array) {
        self.data = DTypes::I8(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithI16Array)]
    pub fn set_data_with_i16_array(&mut self, data: &js_sys::Int16Array) {
        self.data = DTypes::I16(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithI32Array)]
    pub fn set_data_with_i32_array(&mut self, data: &js_sys::Int32Array) {
        self.data = DTypes::I32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU8Array)]
    pub fn set_data_with_u8_array(&mut self, data: &js_sys::Uint8Array) {
        self.data = DTypes::U8(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU16Array)]
    pub fn set_data_with_u16_array(&mut self, data: &js_sys::Uint16Array) {
        self.data = DTypes::U16(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU32Array)]
    pub fn set_data_with_u32_array(&mut self, data: &js_sys::Uint32Array) {
        self.data = DTypes::U32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithF32Array)]
    pub fn set_data_with_f32_array(&mut self, data: &js_sys::Float32Array) {
        self.data = DTypes::F32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithF64Array)]
    pub fn set_data_with_f64_array(&mut self, data: &js_sys::Float64Array) {
        self.data = DTypes::F64(data.to_vec());
    }

    #[wasm_bindgen(js_name = toArray)]
    pub fn to_array(&self) -> js_sys::Array {
        let result = js_sys::Array::new();

        match &self.data {
            DTypes::I8(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::I16(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::I32(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::U8(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::U16(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::U32(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
            DTypes::F32(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from_f64(*v as f64));});
            },
            DTypes::F64(arr) => {
                arr.iter().for_each(|v| {result.push(&JsValue::from(*v));});
            },
        }

        result
    }

    #[wasm_bindgen(js_name=shapeToArray)]
    pub fn shape_to_array(&self) -> js_sys::Array {
        let result = js_sys::Array::new();
        self.shape.iter().for_each(|v| {result.push(&JsValue::from(*v));});

        result
    }
}

impl Tensor {
    pub fn new(data: DTypes, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    #[inline]
    pub fn get_data(&self) -> &DTypes {
        &self.data
    }

    #[inline]
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[inline]
    pub fn get_dim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.shape.iter().fold(1, |res, val| res * val)
    }

    #[inline]
    pub fn set_data(&mut self, data: DTypes) {
        self.data = data;
    }

    #[inline]
    pub fn set_shape(&mut self, shape: Vec<usize>) {
        self.shape = shape.clone();
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> Self {
        self.set_shape(shape);
        (*self).clone()
    }
}

#[wasm_bindgen]
impl TensorList {
    #[wasm_bindgen(constructor)]
    pub fn new_empty() -> Self {
        Self { tensors: Vec::new() }
    }

    pub fn append(&mut self, tensor: &Tensor) {
        self.tensors.push(tensor.clone());
    }
}

impl TensorList {
    pub fn new(tensors: Vec<Tensor>) -> Self {
        Self { tensors }
    }

    #[inline]
    pub fn set(&mut self, tensors: Vec<Tensor>) {
        self.tensors = tensors;
    }

    #[inline]
    pub fn get(&self) -> Vec<Tensor> {
        self.tensors.clone()
    }
}