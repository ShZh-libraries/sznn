import { createBindGroup, getCommandEncoder, getResult, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

export async function handleUnaryOp(input: Tensor, unaryop: string, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);

    const shaderModule = device.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, write> output: array<f32>;
            @group(0) @binding(2) var<storage, read> len : u32;

            let workgroup_size_x = 256;

            @stage(compute) 
            @workgroup_size(workgroup_size_x)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                for (var i: u32 = global_id.x; i < len; i += u32(workgroup_size_x)) {
                    ${unaryop};
                }
            }
        `
    });
    const computePipeline = device.createComputePipeline({
        compute: {
            module: shaderModule,
            entryPoint: "main"
        }
    });

    const bindGroup = createBindGroup(computePipeline, [gpuInputBuffer, gpuOutputBuffer, gpuMetaBuffer], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}
