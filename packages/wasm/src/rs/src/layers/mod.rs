pub mod batchnorm;
pub mod binaryop;
pub mod cast;
pub mod concat;
pub mod conv;
pub mod dropout;
pub mod gather;
pub mod instancenorm;
pub mod padding;
pub mod pooling;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod slice;
pub mod unaryop;
pub mod unsqueeze;
pub mod upsample;

macro_rules! extract_data {
    ($tensor: expr, $dtype: path) => {
        if let $dtype(data) = $tensor.get_data() {
            data
        } else {
            panic!("The data types of tensors do not match!")
        }
    };
}

pub(crate) use extract_data;
