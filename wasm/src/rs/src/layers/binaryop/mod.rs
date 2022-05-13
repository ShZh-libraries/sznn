mod add;
mod sub;
mod mul;
mod div;

fn idx_to_loc(index: usize, stride: &Vec<usize>) -> Vec<usize> {
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

fn loc_to_idx(loc: &Vec<usize>, stride: &Vec<usize>) -> usize {
    let mut index = 0;

    for i in 0..stride.len() {
        index += loc[i] * stride[i];
    }

    index
}

fn get_broadcast_shape(shape1: &Vec<usize>, shape2: &Vec<usize>) -> Vec<usize> {
    let mut broadcast_shape: Vec<usize> = Vec::new();
    let broadcast_len = if shape1.len() > shape2.len() { shape1.len() } else { shape2.len() };

    for index in 0..broadcast_len {
        let a = if shape1.len() - 1 >= index {
            shape1[shape1.len() - 1 - index]
        } else { 1 };
        let b = if shape2.len() - 1 >= index {
            shape2[shape2.len() - 1 - index]
        } else { 1 };

        if a == 1 {
            broadcast_shape.insert(0, b);
        } else if b == 1 {
            broadcast_shape.insert(0, a);
        } else if a == b {
            broadcast_shape.insert(0, a);
        } else {
            panic!("Cannot broadcast!!");
        }
    }

    broadcast_shape
}

fn get_broadcast_dims(shape: &Vec<usize>, broadcast_shape: &Vec<usize>) -> Vec<usize> {
    let mut broadcast_dims: Vec<usize> = Vec::new();
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

fn get_broadcast_loc(out_loc: &Vec<usize>, dim: usize, broadcast_dim: &Vec<usize>) -> Vec<usize> {
    let mut loc = out_loc[(out_loc.len() - dim)..].into_iter().cloned().collect::<Vec<_>>();
    broadcast_dim.iter().for_each(|&dim| loc[dim] = 0);

    loc
}