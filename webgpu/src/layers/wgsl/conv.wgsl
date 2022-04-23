struct PoolingAttr {
    kernel_channel: u32, kernel_height: u32, kernel_width: u32,
    stride_y: u32, stride_x: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> in_shape: vec4<u32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<storage, read> out_shape: vec4<u32>;
@group(0) @binding(4) var<storage, read> weight: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read> attr: PoolingAttr;

let workgroup_size_x = 8;
let workgroup_size_y = 8;
let workgroup_size_z = 4;

@stage(compute)
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn conv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_c = global_id.z;

    if (out_x >= out_shape[3] || out_y >= out_shape[2] || out_c >= out_shape[1]) {
        return;
    }

    let in_start_x = out_x * attr.stride_x;
    let in_start_y = out_y * attr.stride_y;
    let in_c = out_c;

    let kernel_channel_size = attr.kernel_height * attr.kernel_width;
    let kernel_size = attr.kernel_channel * kernel_channel_size;

    var sum = 0.;
    for (var ky = 0u; ky < attr.kernel_height; ky++) {
        for (var kx = 0u; kx < attr.kernel_width; kx++) {
            let in_x = in_start_x + kx;
            let in_y = in_start_y + ky;

            for (var kc = 0u; kc < attr.kernel_channel; kc++) {
                let kernel_idx = out_c * kernel_size + kc * kernel_channel_size + ky * attr.kernel_width + kx;
                let in_idx = kc * in_shape[2] * in_shape[3] + in_y * in_shape[3] + in_x;

                sum += weight[kernel_idx] * input[in_idx];
            }
        }
    }

    let out_idx = out_c * out_shape[2] * out_shape[3] + out_y * out_shape[3] + out_x;
    output[out_idx] = sum + bias[out_c];
}
