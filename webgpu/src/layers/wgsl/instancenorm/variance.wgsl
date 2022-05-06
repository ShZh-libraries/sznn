@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> means: array<f32>;
@group(0) @binding(2) var<storage, write> variance: array<f32>;
@group(0) @binding(3) var<storage, read> channel_num: u32;
@group(0) @binding(4) var<storage, read> channel_size: u32;

let workgroup_size = 256;

fn get_var(offset: u32, len: u32, mean: f32) -> f32 {
    var var_sum: f32 = 0.;

    for (var i = offset; i < offset + len; i++) {
        var_sum += (input[i] - mean) * (input[i] - mean);
    }

    return var_sum / f32(len);
}

@stage(compute)
@workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= channel_num) {
        return;
    }

    let offset = global_id.x * channel_size;
    variance[global_id.x] = get_var(offset, channel_size, means[global_id.x]);
}
