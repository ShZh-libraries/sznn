@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<storage, write> output: array<f32>;
@group(0) @binding(6) var<storage, read> len : u32;

let workgroup_size_x = 256;

@stage(compute) 
@workgroup_size(workgroup_size_x)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    for (var i: u32 = global_id.x; i < len; i += u32(workgroup_size_x)) {
        output[i] = (data[i] - mean[i]) / sqrt(variance[i]) * scale[i] + bias[i];
    }
}