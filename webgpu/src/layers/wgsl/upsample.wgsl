@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> input_stride: vec4<u32>;
@group(0) @binding(2) var<storage, read> scale: vec4<f32>;
@group(0) @binding(3) var<storage, write> output: array<f32>;
@group(0) @binding(4) var<storage, read> output_stride: vec4<u32>;

let workgroup_size_x = 256;

fn idx_to_loc(index: u32, stride: vec4<u32>) -> vec4<u32> {
    var loc = vec4<u32>();
    var idx = index;

    for (var i: u32 = 0u; i < 4u; i++) {
        if (stride[i] != 0u) {
            loc[i] = idx / stride[i];
            idx %= stride[i];
        }
    }

    return loc;
}

fn loc_to_idx(loc: vec4<u32>, stride: vec4<u32>) -> u32 {
    var index: u32 = 0u;

    for (var i: u32 = 0u; i < 4u; i++) {
        index += loc[i] * stride[i];
    }

    return index;
}

@stage(compute)
@workgroup_size(workgroup_size_x)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let out_loc = idx_to_loc(global_id.x, output_stride);

    var in_loc = vec4<u32>();
    for (var i: u32 = 0u; i < 4u; i++) {
        in_loc[i] = out_loc[i] / u32(scale[i]);
    }

    let in_idx = loc_to_idx(in_loc, input_stride);
    output[global_id.x] = input[in_idx];
}