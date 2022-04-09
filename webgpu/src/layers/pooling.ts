import pooling from "./wgsl/pooling.wgsl";
import { DType, Tensor, TensorBuilder } from "../tensor";
import { PoolingAttr } from "../../../core/attr/pooling";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { handlePadding } from "./padding";

export async function handleMaxPool2D(input: Tensor, attr: PoolingAttr, device: GPUDevice): Promise<Tensor> {
  // Padding first
  const paddingAttr = attr.getPaddingAttr();
  let paddingTensor: Tensor;
  if (!paddingAttr.pads.every(x => x == 0)) {
    paddingTensor = await handlePadding(input, paddingAttr, device);
  } else {
    paddingTensor = input;
  }
  
  const maxY = paddingTensor.shape[2] - attr.kernelShape[0];
  const maxX = paddingTensor.shape[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    paddingTensor.shape[0],
    paddingTensor.shape[1],
    outputHeight,
    outputWidth,
  ]);

  const gpuInputBuffer = paddingTensor.setInputGPUBuffer(device);
  const gpuInShapeBuffer = setGPUReadBuffer(new Uint32Array(paddingTensor.shape), DType.uint32, device);
  const gpuOutputBuffer = output.setOutputGPUBuffer(device);
  const gpuOutShapeBuffer = setGPUReadBuffer(new Uint32Array(output.shape), DType.uint32, device);
  const gpuAttrBuffer = setGPUReadBuffer(new Uint32Array([
      attr.kernelShape[0], attr.kernelShape[1],
      attr.strides[0], attr.strides[1]
  ]), DType.uint32, device);

  const computePipeline = loadWGSL(pooling, device, "max_pool");
  const bindGroup = createBindGroup(computePipeline, [
      gpuInputBuffer, gpuInShapeBuffer, 
      gpuOutputBuffer, gpuOutShapeBuffer,
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
  
  // export function handleAvgPool2D(input: Tensor, attr: PoolingAttr): Tensor {
  //   const kernelSize = attr.kernelShape[0] * attr.kernelShape[1];
  //   // Calculate shape
  //   const maxY =
  //     input.shape[2] + attr.pads[0] + attr.pads[2] - attr.kernelShape[0];
  //   const maxX =
  //     input.shape[3] + attr.pads[1] + attr.pads[3] - attr.kernelShape[1];
  //   const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  //   const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  //   const output = TensorBuilder.withShape([
  //     input.shape[0],
  //     input.shape[1],
  //     outputHeight,
  //     outputWidth,
  //   ]);
  
  //   const inputChannelSize = input.shape[2] * input.shape[3];
  //   const inputSize = input.shape[1] * inputChannelSize;
  
  //   let outputIndex = 0;
  //   for (let n = 0; n < input.shape[0]; n++) {
  //     for (let c = 0; c < input.shape[1]; c++) {
  //       for (
  //         let y = -attr.pads[0];
  //         y <= maxY - attr.pads[0];
  //         y += attr.strides[0]
  //       ) {
  //         for (
  //           let x = -attr.pads[1];
  //           x <= maxX - attr.pads[1];
  //           x += attr.strides[1]
  //         ) {
  //           let sum = 0;
  //           for (let ky = 0; ky < attr.kernelShape[0]; ky++) {
  //             for (let kx = 0; kx < attr.kernelShape[1]; kx++) {
  //               const cy = y + ky;
  //               const cx = x + kx;
  //               if (
  //                 cy >= 0 &&
  //                 cy < input.shape[2] &&
  //                 cx >= 0 &&
  //                 cx < input.shape[3]
  //               ) {
  //                 const curIdx =
  //                   n * inputSize +
  //                   c * inputChannelSize +
  //                   cy * input.shape[3] +
  //                   cx;
  //                 sum += input.data[curIdx];
  //               }
  //             }
  //           }
  //           output.data[outputIndex++] = sum / kernelSize;
  //         }
  //       }
  //     }
  //   }
  
  //   return output;
  // }
  
  // export function handleGlobalAvgPool(input: Tensor): Tensor {
  //   const globalAvgPoolingAttr = new PoolingAttr();
  //   globalAvgPoolingAttr.kernelShape = [input.shape[2], input.shape[3]];
  //   const output = handleAvgPool2D(input, globalAvgPoolingAttr);
  
  //   return output;
  // }
  