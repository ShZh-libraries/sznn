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
pub enum TensorDataType {
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Tensor {
    #[wasm_bindgen(skip)]
    pub data: TensorDataType,
    shape: Vec<usize>,
    pub ndim: usize,
    pub length: usize,
}

#[wasm_bindgen]
impl Tensor {
    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape(&mut self, shape: &js_sys::Array) {
        let mut tensor_shape = Vec::with_capacity(shape.length() as usize);
        let mut tensor_length = 1;

        for num in shape.iter() {
            let dim = num.as_f64().unwrap() as usize;
            tensor_shape.push(dim);
            tensor_length *= dim;
        }

        self.shape = tensor_shape;
        self.length = tensor_length;
        self.ndim = shape.length() as usize;
    }

    #[wasm_bindgen(js_name = setDataWithArray)]
    pub fn set_data_with_array(&mut self, data: &js_sys::Array, dtype: DType) {
        match dtype {
            DType::Int8 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i8;
                });
                self.data = TensorDataType::Int8(tensor_data);
            },
            DType::Int16 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i16;
                });
                self.data = TensorDataType::Int16(tensor_data);
            },
            DType::Int32 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as i32;
                });
                self.data = TensorDataType::Int32(tensor_data);
            },
            DType::UInt8 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u8;
                });
                self.data = TensorDataType::UInt8(tensor_data);
            },
            DType::UInt16 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u16;
                });
                self.data = TensorDataType::UInt16(tensor_data);
            },
            DType::UInt32 => {
                let mut tensor_data = vec![0; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as u32;
                });
                self.data = TensorDataType::UInt32(tensor_data);
            },
            DType::Float32 => {
                let mut tensor_data = vec![0.; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap() as f32;
                });
                self.data = TensorDataType::Float32(tensor_data);
            },
            DType::Float64 => {
                let mut tensor_data = vec![0.; data.length() as usize];
                data.for_each(&mut |value: JsValue, index: u32, _| {
                    tensor_data[index as usize] = value.as_f64().unwrap();
                });
                self.data = TensorDataType::Float64(tensor_data);
            },
        }
        
    }

    #[wasm_bindgen(js_name = setDataWithI8Array)]
    pub fn set_data_with_i8_array(&mut self, data: &js_sys::Int8Array) {
        self.data = TensorDataType::Int8(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithI16Array)]
    pub fn set_data_with_i16_array(&mut self, data: &js_sys::Int16Array) {
        self.data = TensorDataType::Int16(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithI32Array)]
    pub fn set_data_with_i32_array(&mut self, data: &js_sys::Int32Array) {
        self.data = TensorDataType::Int32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU8Array)]
    pub fn set_data_with_u8_array(&mut self, data: &js_sys::Uint8Array) {
        self.data = TensorDataType::UInt8(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU16Array)]
    pub fn set_data_with_u16_array(&mut self, data: &js_sys::Uint16Array) {
        self.data = TensorDataType::UInt16(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithU32Array)]
    pub fn set_data_with_u32_array(&mut self, data: &js_sys::Uint32Array) {
        self.data = TensorDataType::UInt32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithF32Array)]
    pub fn set_data_with_f32_array(&mut self, data: &js_sys::Float32Array) {
        self.data = TensorDataType::Float32(data.to_vec());
    }

    #[wasm_bindgen(js_name = setDataWithF64Array)]
    pub fn set_data_with_f64_array(&mut self, data: &js_sys::Float64Array) {
        self.data = TensorDataType::Float64(data.to_vec());
    }
}

impl Tensor {
    pub fn new() -> Self {
        Self {
            data: TensorDataType::Int8(Vec::new()),
            shape: Vec::new(),
            ndim: 0,
            length: 0,
        }
    }

    #[inline]
    pub fn get_ndim(&self) -> usize {
        self.ndim
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn set_vec_shape(&mut self, shape: Vec<usize>) {
        self.ndim = shape.len();
        self.length = shape.iter().fold(1, |res, val| res * val );
        self.shape = shape.clone();
    }

    pub fn with_shape(shape: Vec<usize>, dtype: Option<DType>) -> Tensor {
        let dtype = dtype.unwrap_or_else(|| DType::Float64);
        let ndim = shape.len();
        let length = shape.iter().fold(1, |res, val| res * val );
        let data = match dtype {
            DType::Int8 => TensorDataType::Int8(vec![0; length]),
            DType::Int16 => TensorDataType::Int16(vec![0; length]),
            DType::Int32 => TensorDataType::Int32(vec![0; length]),
            DType::UInt8 => TensorDataType::UInt8(vec![0; length]),
            DType::UInt16 => TensorDataType::UInt16(vec![0; length]),
            DType::UInt32 => TensorDataType::UInt32(vec![0; length]),
            DType::Float32 => TensorDataType::Float32(vec![0.; length]),
            DType::Float64 => TensorDataType::Float64(vec![0.; length]),
        };

        Tensor {
            data,
            shape,
            ndim,
            length,
        }
    }
}
