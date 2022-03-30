import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class PoolingAttr {
  kernelShape: number[] = [];
  pads: number[] = [0, 0, 0, 0];
  strides: number[] = [1, 1];
}

function handleAttribute(attributes: onnx.AttributeProto[]): PoolingAttr {
  let result: PoolingAttr = new PoolingAttr();
  for (const attribute of attributes) {
    switch (attribute.name) {
      case "kernel_shape":
        result.kernelShape = attribute.ints as number[];
        break;
      case "pads":
        result.pads = attribute.ints as number[];
        break;
      case "strides":
        result.strides = attribute.ints as number[];
        break;
      default:
        throw new Error(
          `Unknown attribute ${attribute.name} in MaxPooling layer!!`
        );
    }
  }

  return result;
}

export function handleMaxPool2D(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const maxPoolingAttr = handleAttribute(attributes);
  const output = forwardMaxPool2D(inputs[0], maxPoolingAttr);

  return [output];
}

export function handleAvgPool2D(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const avgPoolingAttr = handleAttribute(attributes);
  const output = forwardAvgPool2D(inputs[0], avgPoolingAttr);

  return [output];
}

export function handleGlobalAvgPool(inputs: Tensor[]): Tensor[] {
  const globalAvgPoolingAttr = new PoolingAttr();
  globalAvgPoolingAttr.kernelShape = [inputs[0].shape[2], inputs[0].shape[3]];
  const output = forwardAvgPool2D(inputs[0], globalAvgPoolingAttr);

  return [output];
}

export function forwardMaxPool2D(input: Tensor, attr: PoolingAttr): Tensor {
  // Calculate shape
  const maxY =
    input.shape[2] + attr.pads[0] + attr.pads[2] - attr.kernelShape[0];
  const maxX =
    input.shape[3] + attr.pads[1] + attr.pads[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    input.shape[0],
    input.shape[1],
    outputHeight,
    outputWidth,
  ]);

  const inputChannelSize = input.shape[2] * input.shape[3];
  const inputSize = input.shape[1] * inputChannelSize;

  let outputIndex = 0;
  for (let n = 0; n < input.shape[0]; n++) {
    for (let c = 0; c < input.shape[1]; c++) {
      for (
        let y = -attr.pads[0];
        y <= maxY - attr.pads[0];
        y += attr.strides[0]
      ) {
        for (
          let x = -attr.pads[1];
          x <= maxX - attr.pads[1];
          x += attr.strides[1]
        ) {
          let maxIdx = n * inputSize + c * inputChannelSize + y * input.shape[3] + x;
          let maxValue = input.data[maxIdx];
          for (let ky = 0; ky < attr.kernelShape[0]; ky++) {
            // Kernel
            for (let kx = 0; kx < attr.kernelShape[1]; kx++) {
              const cy = y + ky; // Current
              const cx = x + kx;
              let currentValue = 0;
              if (
                cy >= 0 &&
                cy < input.shape[2] &&
                cx >= 0 &&
                cx < input.shape[3]
              ) {
                const currentIdx = n * inputSize + c * inputChannelSize + cy * input.shape[3] + cx;
                currentValue = input.data[currentIdx];
              }
              if (currentValue > maxValue) {
                maxValue = currentValue;
              }
            }
          }

          output.data[outputIndex++] = maxValue;
        }
      }
    }
  }

  return output;
}

export function forwardAvgPool2D(input: Tensor, attr: PoolingAttr): Tensor {
  const kernelSize = attr.kernelShape[0] * attr.kernelShape[1];
  // Calculate shape
  const maxY =
    input.shape[2] + attr.pads[0] + attr.pads[2] - attr.kernelShape[0];
  const maxX =
    input.shape[3] + attr.pads[1] + attr.pads[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    input.shape[0],
    input.shape[1],
    outputHeight,
    outputWidth,
  ]);

  const inputChannelSize = input.shape[2] * input.shape[3];
  const inputSize = input.shape[1] * inputChannelSize;

  let outputIndex = 0;
  for (let n = 0; n < input.shape[0]; n++) {
    for (let c = 0; c < input.shape[1]; c++) {
      for (
        let y = -attr.pads[0];
        y <= maxY - attr.pads[0];
        y += attr.strides[0]
      ) {
        for (
          let x = -attr.pads[1];
          x <= maxX - attr.pads[1];
          x += attr.strides[1]
        ) {
          let sum = 0;
          for (let ky = 0; ky < attr.kernelShape[0]; ky++) {
            for (let kx = 0; kx < attr.kernelShape[1]; kx++) {
              const cy = y + ky;
              const cx = x + kx;
              if (
                cy >= 0 &&
                cy < input.shape[2] &&
                cx >= 0 &&
                cx < input.shape[3]
              ) {
                const curIdx = n * inputSize + c * inputChannelSize + cy * input.shape[3] + cx
                sum += input.data[curIdx];
              }
            }
          }
          output.data[outputIndex++] = sum / kernelSize;
        }
      }
    }
  }

  return output;
}
