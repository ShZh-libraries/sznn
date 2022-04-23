import conv from "./wgsl/conv.wgsl";
import { ConvAttr } from "../../../core/attr/conv";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";
import { handlePadding } from "./padding";

export async function handleConv(device: GPUDevice, attr: ConvAttr, input: Tensor, weight: Tensor, bias?: Tensor): Promise<Tensor> {
    // Padding first
    const paddingAttr = attr.getPaddingAttr();
    let paddingTensor: Tensor;
    if (!paddingAttr.pads.every(x => x == 0)) {
      paddingTensor = await handlePadding(input, paddingAttr, device);
    } else {
      paddingTensor = input;
    }
    
    const outputSize = input.shape[0];
    const outputChannel = weight.shape[0];
    const maxY = paddingTensor.shape[2] - attr.kernelShape[0];
    const maxX = paddingTensor.shape[3] - attr.kernelShape[1];
    const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
    const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
    const output = TensorBuilder.withShape([
      outputSize,
      outputChannel,
      outputHeight,
      outputWidth,
    ]);
  
    const gpuInputBuffer = paddingTensor.setInputGPUBuffer(device);
    const gpuInShapeBuffer = setGPUReadBuffer(new Uint32Array(paddingTensor.shape), DType.uint32, device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);
    const gpuOutShapeBuffer = setGPUReadBuffer(new Uint32Array(output.shape), DType.uint32, device);
    const gpuWeightBuffer = weight.setInputGPUBuffer(device);
    let gpuBiasBuffer;
    if (bias) {
        gpuBiasBuffer = bias.setInputGPUBuffer(device);
    } else {
        gpuBiasBuffer = TensorBuilder.withShape([output.shape[1]]).setInputGPUBuffer(device);
    }
    const gpuAttrBuffer = setGPUReadBuffer(new Uint32Array([
        weight.shape[1], attr.kernelShape[0], attr.kernelShape[1],
        attr.strides[0], attr.strides[1]
    ]), DType.uint32, device);
  
    const computePipeline = loadWGSL(conv, device, "conv");
    const bindGroup = createBindGroup(computePipeline, [
        gpuInputBuffer, gpuInShapeBuffer, 
        gpuOutputBuffer, gpuOutShapeBuffer,
        gpuWeightBuffer, gpuBiasBuffer,
        gpuAttrBuffer
    ], device);
    // TODO
    const commandEncoder = getCommandEncoder(
      computePipeline, 
      bindGroup, 
      [Math.ceil(output.shape[3] / 8), Math.ceil(output.shape[2] / 8), Math.ceil(output.shape[1] / 4)], 
      device
    );
  
    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);
    output.data = resultArray;
  
    return output;
  }
  