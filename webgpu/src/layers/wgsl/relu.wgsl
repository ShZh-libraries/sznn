@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;
@group(0) @binding(2) var<storage, read> len: u32;
@group(0) @binding(3) var<storage, read> alpha: f32;

let workgroup_size_x = 256;

@stage(compute) 
@workgroup_size(workgroup_size_x)
fn relu(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= len) {
        return;
    }
    output[global_id.x] = max(input[global_id.x], 0.) * alpha;
}