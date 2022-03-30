import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class ConvAttr {
  autoPad: string = "NOTSET";
  dilations: number[] = [];
  group: number = 1;
  kernelShape: number[] = [];
  pads: number[] = [0, 0, 0, 0];
  strides: number[] = [1, 1];
}

function handleAttribute(attributes: onnx.AttributeProto[]): ConvAttr {
  let result: ConvAttr = new ConvAttr();
  for (const attribute of attributes) {
    switch (attribute.name) {
      case "dilations":
        result.dilations = attribute.ints as number[];
        break;
      case "group":
        result.group = attribute.i as number;
        break;
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
          `Unknown attribute ${attribute.name} in Convolutional layer!!`
        );
    }
  }

  return result;
}

export function handleConv(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const convAttr = handleAttribute(attributes);
  const output = forward(convAttr, inputs[0], inputs[1], inputs[2]);

  return output;
}

export function forward(
  attr: ConvAttr,
  input: Tensor,
  weight: Tensor,
  bias?: Tensor
): Tensor[] {
  // Calculate shape
  const outputSize = input.shape[0];
  const outputChannel = weight.shape[0] / attr.group;

  const maxY = input.shape[2] + attr.pads[0] + attr.pads[2] - weight.shape[2];
  const maxX = input.shape[3] + attr.pads[1] + attr.pads[3] - weight.shape[3];
  const outputHeight = Math.floor(maxY / attr.strides[1]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[0]) + 1;
  const outputShape = [outputSize, outputChannel, outputHeight, outputWidth];
  const output = TensorBuilder.withShape(outputShape);

  let resultIndex = 0;
  for (let n = 0; n < outputSize; n++) {
    for (let c = 0; c < outputChannel; c++) {
      for (
        let y = -attr.pads[0];
        y <= maxY - attr.pads[0];
        y += attr.strides[1]
      ) {
        for (
          let x = -attr.pads[1];
          x <= maxX - attr.pads[1];
          x += attr.strides[0]
        ) {
          let sum = 0;
          for (let ky = 0; ky < attr.kernelShape[1]; ky++) {
            for (let kx = 0; kx < attr.kernelShape[0]; kx++) {
              const cy = y + ky;
              const cx = x + kx;

              if (
                cx >= 0 &&
                cx < input.shape[3] &&
                cy >= 0 &&
                cy < input.shape[2]
              ) {
                for (let kc = 0; kc < weight.shape[1]; kc++) {
                  const kernelValue = weight.atLoc([c, kc, ky, kx]);
                  const dataValue = input.atLoc([n, kc, cy, cx]);

                  sum += kernelValue * dataValue;
                }
              }
            }
          }
          output.data[resultIndex++] = sum + (bias ? bias.data[c] : 0);
        }
      }
    }
  }

  return [output];
}

// export function forward(data: Tensor, weight: Tensor, convAttr: ConvAttr): Tensor[] {
//     // Calculate shape
//     const resultSize = data.shape[0];
//     const resultChannel = weight.shape[0];
//     const kernelWidth = weight.shape[2];
//     const kernelHeight = weight.shape[3];

//     const resultWidth = Math.floor(data.shape[2] + convAttr.pads[0] + convAttr.pads[2] - kernelWidth) / convAttr.strides[0] + 1;
//     const resultHeight = Math.floor(data.shape[3] + convAttr.pads[1] + convAttr.pads[3] - kernelHeight) / convAttr.strides[1] + 1;
//     const resultShape = [resultSize, resultChannel, resultWidth, resultHeight];
//     const result = TensorBuilder.withShape(resultShape);

//     // Do convolution
//     const kernelSize = kernelWidth * kernelHeight;
//     for (let y = 0; y < resultHeight; y++) {
//         const tempTensor = im2row(data, convAttr, y * convAttr.strides[1]);
//         for (let x = 0; x < resultWidth; x++) {
//             for (let c = 0; c < resultChannel; c++) {
//                 let sum = 0;
//                 for (let i = 0; i < kernelSize; i++) {
//                     sum += weight.data[c * kernelSize + i] as number
//                 }
//             }
//         }
//     }

//     return [result];
// }

// export function im2row(data: Tensor, convAttr: ConvAttr, y: number): Tensor {
//     const channel = data.shape[1];
//     const width = data.shape[2];

//     for (let i = 0; i < channel; i++) {
//         for (let j = 0; j < convAttr.kernelShape[1]; j++) {
//             for (let x = 0; x < width; x += convAttr.strides[0]) {
//                 for (let k = 0; k < convAttr.kernelShape[0]; x++) {
//                     const srcX = x - convAttr.pads[0] + k;
//                     const srcY = y - convAttr.pads[1] + j;

//                     let target = 0;
//                     if (srcX > 0 && srcY > 0 && srcX < data.shape[2] && srcY < data.shape[3]) {
//                         target = data.data[1] as number;
//                     }
//                 }
//             }
//         }
//     }
// }
