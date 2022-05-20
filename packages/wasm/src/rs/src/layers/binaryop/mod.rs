mod add;
mod div;
mod mul;
mod sub;

macro_rules! par {
    ($a: expr, $b: expr, $typ: ty) => {{
        let mut out = vec![0 as $typ; $a.len()];
        $a.par_iter()
            .zip($b.par_iter())
            .zip(out.par_iter_mut())
            .for_each(|((a, b), dst)| {
                *dst = a * b;
            });
        out
    }};
}

macro_rules! simd_par {
    ($a: expr, $b: expr, $typ: ty, $simd: ident) => {{
        let chunks_num: usize = 16 / mem::size_of::<$typ>();
        let mut out = vec![0 as $typ; $a.len()];
        $a.par_chunks(chunks_num)
            .zip($b.par_chunks(chunks_num))
            .zip(out.par_chunks_mut(chunks_num))
            .for_each(|((a, b), dst)| unsafe {
                let a = v128_load(a.as_ptr() as *const v128);
                let b = v128_load(b.as_ptr() as *const v128);
                let result = $simd(a, b);
                v128_store(dst.as_mut_ptr() as *mut v128, result);
            });
        out
    }};
}

macro_rules! broadcast_inner {
    (
        $a: expr, $b: expr, $typ: ty, $op: tt,
        $a_stride: expr, $a_dim: expr,
        $b_stride: expr, $b_dim: expr,
        $out_stride: expr, $out_len: expr,
        $a_broadcast_dim: expr, $b_broadcast_dim: expr
    ) => {
        {
            let mut out_data = vec![0 as $typ; $out_len];
            let out_data_mutex = Mutex::new(&mut out_data);

            (0..$out_len).into_par_iter().for_each(|i| {
                let out_loc = idx_to_loc(i, &$out_stride);

                let a_loc = get_broadcast_loc(&out_loc, $a_dim, $a_broadcast_dim);
                let a_idx = loc_to_idx(&a_loc, &$a_stride);

                let b_loc = get_broadcast_loc(&out_loc, $b_dim, $b_broadcast_dim);
                let b_idx = loc_to_idx(&b_loc, &$b_stride);

                let value = $a[a_idx] $op $b[b_idx];
                let mut out_data = out_data_mutex.lock().unwrap();
                out_data[i] = value;
            });

            out_data
        }
    };
}

macro_rules! broadcast {
    ($name: ident, $op: tt) => {
        paste! {
            fn [< $name _broadcast >](a: &Tensor, b: &Tensor, a_broadcast_dim: &[usize], b_broadcast_dim: &[usize], out_shape: Vec<usize>) -> Tensor {
                let mut output = Tensor::new_empty();
                output.set_shape(out_shape);

                let a_stride = a.get_stride();
                let a_dim = a_stride.len();
                let b_stride = b.get_stride();
                let b_dim = b_stride.len();
                let out_len = output.get_length();
                let out_stride = output.get_stride();

                let out_data = match &a.get_data() {
                    DTypes::I8(a) => {
                        let b = extract_data!(b, DTypes::I8);
                        DTypes::I8(broadcast_inner!(
                            a, b, i8, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    }
                    DTypes::I16(a) => {
                        let b = extract_data!(b, DTypes::I16);
                        DTypes::I16(broadcast_inner!(
                            a, b, i16, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::I32(a) => {
                        let b = extract_data!(b, DTypes::I32);
                        DTypes::I32(broadcast_inner!(
                            a, b, i32, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::U8(a) => {
                        let b = extract_data!(b, DTypes::U8);
                        DTypes::U8(broadcast_inner!(
                            a, b, u8, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::U16(a) => {
                        let b = extract_data!(b, DTypes::U16);
                        DTypes::U16(broadcast_inner!(
                            a, b, u16, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::U32(a) => {
                        let b = extract_data!(b, DTypes::U32);
                        DTypes::U32(broadcast_inner!(
                            a, b, u32, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::F32(a) => {
                        let b = extract_data!(b, DTypes::F32);
                        DTypes::F32(broadcast_inner!(
                            a, b, f32, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                    DTypes::F64(a) => {
                        let b = extract_data!(b, DTypes::F64);
                        DTypes::F64(broadcast_inner!(
                            a, b, f64, $op,
                            a_stride, a_dim,
                            b_stride, b_dim,
                            out_stride, out_len,
                            a_broadcast_dim, b_broadcast_dim
                        ))
                    },
                };

                output.set_data(out_data);

                output
            }
        }
    };
}

macro_rules! handle_binaryop {
    ($name: expr, $op: tt) => {
        paste! {
            #[wasm_bindgen(js_name = [<handle $name:camel>])]
            pub fn [<handle_ $name>](a: &Tensor, b: &Tensor) -> Tensor {
                let (a_shape, b_shape) = (a.get_shape(), b.get_shape());
                let out_shape = get_broadcast_shape(&a_shape, &b_shape);
                let (a_broadcast_dim, b_broadcast_dim) = (get_broadcast_dims(&a_shape, &out_shape), get_broadcast_dims(&b_shape, &out_shape));

                if a_broadcast_dim.len() + b_broadcast_dim.len() == 0 {
                    [<$name _same_shape>](a, b, out_shape)
                } else {
                    [<$name _broadcast>](a, b, &a_broadcast_dim, &b_broadcast_dim, out_shape)
                }
            }
        }
    };
}

fn idx_to_loc(index: usize, stride: &[usize]) -> Vec<usize> {
    let mut loc = vec![0; stride.len()];
    let mut index = index;

    for i in 0..stride.len() {
        if stride[i] != 0 {
            loc[i] = index / stride[i];
            index %= stride[i];
        }
    }

    loc
}

fn loc_to_idx(loc: &[usize], stride: &[usize]) -> usize {
    let mut index = 0;

    for i in 0..stride.len() {
        index += loc[i] * stride[i];
    }

    index
}

fn get_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let mut broadcast_shape: Vec<usize> = Vec::new();
    let broadcast_len = if shape1.len() > shape2.len() {
        shape1.len()
    } else {
        shape2.len()
    };

    for index in 0..broadcast_len {
        let a = if shape1.len() > index {
            shape1[shape1.len() - 1 - index]
        } else {
            1
        };
        let b = if shape2.len() > index {
            shape2[shape2.len() - 1 - index]
        } else {
            1
        };

        if a == 1 {
            broadcast_shape.insert(0, b);
        } else if b == 1 || a == b {
            broadcast_shape.insert(0, a);
        } else {
            panic!("Cannot broadcast!!");
        }
    }

    broadcast_shape
}

fn get_broadcast_dims(shape: &[usize], broadcast_shape: &[usize]) -> Vec<usize> {
    let mut broadcast_dims = Vec::new();
    for index in 0..shape.len() {
        let dim = shape.len() - 1 - index;
        let a = shape[dim];
        let b = broadcast_shape[dim];
        if b > 1 && a == 1 {
            broadcast_dims.insert(0, dim)
        }
    }

    broadcast_dims
}

fn get_broadcast_loc(out_loc: &[usize], dim: usize, broadcast_dim: &[usize]) -> Vec<usize> {
    let mut loc = out_loc[(out_loc.len() - dim)..].to_vec();
    broadcast_dim.iter().for_each(|&dim| loc[dim] = 0);

    loc
}

pub(crate) use broadcast;
pub(crate) use broadcast_inner;
pub(crate) use handle_binaryop;
pub(crate) use par;
pub(crate) use simd_par;
