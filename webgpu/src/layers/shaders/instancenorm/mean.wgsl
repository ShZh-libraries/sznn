@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> means: array<f32>;
@group(0) @binding(2) var<storage, read> channel_num: u32;
@group(0) @binding(3) var<storage, read> channel_size: u32;

let workgroup_size = 256;

fn get_mean(offset: u32, len: u32) -> f32 {
    var sum: f32 = 0.;

    for (var i = offset; i < offset + len; i++) {
        sum += input[i];
    }

    return sum / f32(len);
}

@stage(compute)
@workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= channel_num) {
        return;
    }

    let offset = global_id.x * channel_size;
    means[global_id.x] = get_mean(offset, channel_size);
}
