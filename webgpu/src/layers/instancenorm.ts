import instancenorm from "./wgsl/instancenorm.wgsl";
import { DType, Tensor, TensorBuilder } from "../tensor";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";

export async function handleInstanceNorm(input: Tensor, weight: Tensor, bias: Tensor, epsilon: number, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuWeightBuffer = weight.setInputGPUBuffer(device);
    const gpuBiasBuffer = bias.setInputGPUBuffer(device);
    const gpuEpsilonBuffer = setGPUReadBuffer(new Float32Array([epsilon]), DType.float32, device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);
    const gpuChannelBuffer = setGPUReadBuffer(new Uint32Array([output.shape[1], output.shape[2] * output.shape[3]]), DType.uint32, device);
    
    const computePipeline = loadWGSL(instancenorm, device);

    const bindGroup = createBindGroup(computePipeline, [
        gpuInputBuffer, 
        gpuWeightBuffer,
        gpuBiasBuffer,
        gpuEpsilonBuffer,
        gpuOutputBuffer,
        gpuChannelBuffer,
    ], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}