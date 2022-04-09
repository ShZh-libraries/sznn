import relu from "./wgsl/relu.wgsl";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

export async function handleRelu(input: Tensor, device: GPUDevice): Promise<Tensor> {
    let output = TensorBuilder.withShape(input.shape);

    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);

    const computePipeline = loadWGSL(relu, device, "relu");

    const bindGroup = createBindGroup(computePipeline, [gpuInputBuffer, gpuOutputBuffer, gpuMetaBuffer], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

