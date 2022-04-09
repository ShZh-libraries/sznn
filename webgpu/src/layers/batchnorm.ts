import batchnorm from "./wgsl/batchnorm.wgsl";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

export async function handleBatchNorm(
    data: Tensor,
    scale: Tensor,
    bias: Tensor,
    mean: Tensor,
    variance: Tensor,
    device: GPUDevice,
): Promise<Tensor> {
    let output = TensorBuilder.withShape(data.shape);

    const len = output.getLength();
    const gpuMetaBuffer = setGPUReadBuffer(new Uint32Array([len]), DType.uint32, device);
    const gpuDataBuffer = data.setInputGPUBuffer(device);
    const gpuScaleBuffer = scale.setInputGPUBuffer(device);
    const gpuBiasBuffer = bias.setInputGPUBuffer(device);
    const gpuMeanBuffer = mean.setInputGPUBuffer(device);
    const gpuVarianceBuffer = variance.setInputGPUBuffer(device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);

    const computePipeline = loadWGSL(batchnorm, device);

    const bindGroup = createBindGroup(computePipeline, [
        gpuDataBuffer,
        gpuScaleBuffer,
        gpuBiasBuffer,
        gpuMeanBuffer,
        gpuVarianceBuffer,
        gpuOutputBuffer, 
        gpuMetaBuffer
    ], device);

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

