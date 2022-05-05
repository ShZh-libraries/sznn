@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> means: array<f32>;
@group(0) @binding(2) var<storage, write> temp: array<f32>;
@group(0) @binding(3) var<storage, write> variance: array<f32>;
@group(0) @binding(4) var<storage, read> channel_num: u32;
@group(0) @binding(5) var<storage, read> channel_size: u32;

let mean_workgroup_size_x = 8;
let mean_workgroup_size_y = 32;

fn get_var(offset: u32, len: u32, tid: u32, mean: f32) -> f32 {
    var var_sum: f32;

    if (tid < len / 2u) {
        let std_var1 = input[offset + tid] - mean;
        let std_var2 = input[offset + tid + len / 2u] - mean;
        temp[offset + tid] = std_var1 * std_var1 + std_var2 * std_var2;
    }
    workgroupBarrier();

    for (var stride: u32 = len / 4u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            temp[offset + tid] += temp[offset + tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (len % 2u == 0u) {
            var_sum = temp[offset];
        } else {
            let std_var = input[offset + len - 1u] - mean;
            var_sum = temp[offset] + std_var * std_var;
        }
    }
    workgroupBarrier();

    return var_sum / f32(len);
}

@stage(compute)
@workgroup_size(mean_workgroup_size_x, mean_workgroup_size_y)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= channel_num || global_id.y >= channel_size) {
        return;
    }

    let offset = global_id.x * channel_size;
    variance[global_id.x] = get_var(offset, channel_size, global_id.y, means[global_id.x]);
}
