struct Channel {
    num: u32,
    size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> epsilon: f32;
@group(0) @binding(4) var<storage, write> output: array<f32>;
@group(0) @binding(5) var<storage, read> channel: Channel;

let workgroup_size_x = 256;

var<workgroup> sum: f32;
var<workgroup> var_sum: f32;

fn get_mean(offset: u32, len: u32, tid: u32) -> f32 {
    if (tid < len / 2u) {
        output[offset + tid] = input[offset + tid] + input[offset + tid + len / 2u];
    }
    workgroupBarrier();

    for (var stride: u32 = len / 4u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            output[offset + tid] += output[offset + tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (len % 2u == 0u) {
            sum = output[offset];
        } else { 
            sum = output[offset] + input[offset + len - 1u];
        }
    }
    workgroupBarrier();

    return sum / f32(len);
}

fn get_var(offset: u32, len: u32, tid: u32, mean: f32) -> f32 {
    if (tid < len / 2u) {
        let std_var1 = input[offset + tid] - mean;
        let std_var2 = input[offset + tid + len / 2u] - mean;
        output[offset + tid] = std_var1 * std_var1 + std_var2 * std_var2;
    }
    workgroupBarrier();

    for (var stride: u32 = len / 4u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            output[offset + tid] += output[offset + tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (len % 2u == 0u) {
            var_sum = output[offset];
        } else {
            let std_var = input[offset + len - 1u] - mean;
            var_sum = output[offset] + std_var * std_var;
        }
    }
    workgroupBarrier();

    return var_sum / f32(len);
}

@stage(compute) 
@workgroup_size(workgroup_size_x)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    for (var c: u32 = 0u; c < channel.num; c++) {
        let offset = c * channel.size;

        let mean = get_mean(offset, channel.size, global_id.x);
        let variance = get_var(offset, channel.size, global_id.x, mean);
        
        for (var i: u32 = global_id.x; i < channel.size; i += u32(workgroup_size_x)) {
            output[offset + i] = weight[c] * (input[offset + i] - mean) / sqrt(variance + epsilon) + bias[c];
        }
    }
}
