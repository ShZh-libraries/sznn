struct PaddingAttr {
    pad_top: u32,
    pad_left: u32,
    pad_bottom: u32,
    pad_right: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> in_shape: vec4<u32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<storage, read> out_shape: vec4<u32>;
@group(0) @binding(4) var<storage, read> attr: PaddingAttr;

let workgroup_size_x = 16;
let workgroup_size_y = 16;

// 3D input and 2D padding
@stage(compute)
@workgroup_size(workgroup_size_x, workgroup_size_y)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_c = global_id.z;

    if (out_x >= out_shape[3] || out_y >= out_shape[2] || out_c >= out_shape[1]) {
        return;
    }

    let in_x = out_x - attr.pad_left;
    let in_y = out_y - attr.pad_top;

    let out_idx = out_c * out_shape[2] * out_shape[3]  + out_y * out_shape[3] + out_x;
    if (in_x >= 0u && in_x < in_shape[3] && in_y >= 0u && in_y < in_shape[2]) {
        let in_idx = out_c * in_shape[2] * in_shape[3] + in_y * in_shape[3] + in_x;
        output[out_idx] = input[in_idx];
    } else {
        output[out_idx] = 0.;
    }
}