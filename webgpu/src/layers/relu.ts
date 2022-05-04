import relu from "./wgsl/relu.wgsl";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

const workgroup_size = 256;

export async function handleRelu(input: Tensor, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);
    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuAlphaBuffer = setGPUReadBuffer(new Float32Array([1]), DType.float32, device);

    const computePipeline = loadWGSL(relu, device, "relu");

    const bindGroup = createBindGroup(computePipeline, [
        gpuInputBuffer, 
        gpuOutputBuffer, 
        gpuMetaBuffer,
        gpuAlphaBuffer,
    ], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [Math.ceil(len / workgroup_size)], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

export async function handleLeakyRelu(input: Tensor, alpha: number, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);
    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuAlphaBuffer = setGPUReadBuffer(new Float32Array([alpha]), DType.float32, device);
    
    const computePipeline = loadWGSL(relu, device, "relu");

    const bindGroup = createBindGroup(computePipeline, [
        gpuInputBuffer, 
        gpuOutputBuffer, 
        gpuMetaBuffer,
        gpuAlphaBuffer,
    ], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [Math.ceil(len / workgroup_size)], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

