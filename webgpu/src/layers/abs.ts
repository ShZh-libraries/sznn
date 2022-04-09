import abs from "./wgsl/abs.wgsl";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

export async function handleAbs(input: Tensor, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);

    const computePipeline = loadWGSL(abs, device);

    const bindGroup = createBindGroup(computePipeline, [gpuInputBuffer, gpuOutputBuffer, gpuMetaBuffer], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

