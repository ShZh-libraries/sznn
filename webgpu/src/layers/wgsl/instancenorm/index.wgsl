struct InstanceNormMeta {
    len: u32,
    channel_size: u32
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> means: array<f32>;
@group(0) @binding(4) var<storage, read> variances: array<f32>;
@group(0) @binding(5) var<storage, read> epsilon: f32;
@group(0) @binding(6) var<storage, write> output: array<f32>;
@group(0) @binding(7) var<storage, read> meta: InstanceNormMeta;

let workgroup_size_x = 256;

@stage(compute) 
@workgroup_size(workgroup_size_x)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= meta.len) {
        return;
    }

    let channel = global_id.x / meta.channel_size;
    output[global_id.x] = weight[channel] * (input[global_id.x] - means[channel]) / sqrt(variances[channel] + epsilon) + bias[channel];
}