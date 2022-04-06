import { ConvAttr } from "../../../core/attr/conv";
import { Tensor, TensorBuilder } from "../tensor";

export function handleConv(
  attr: ConvAttr,
  input: Tensor,
  weight: Tensor,
  bias?: Tensor
): Tensor {
  // Calculate shape
  const outputSize = input.shape[0];
  const outputChannel = weight.shape[0] / attr.group;

  const maxY = input.shape[2] + attr.pads[0] + attr.pads[2] - weight.shape[2];
  const maxX = input.shape[3] + attr.pads[1] + attr.pads[3] - weight.shape[3];
  const outputHeight = Math.floor(maxY / attr.strides[1]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[0]) + 1;
  const outputShape = [outputSize, outputChannel, outputHeight, outputWidth];
  const output = TensorBuilder.withShape(outputShape);

  const kernelChannelSize = weight.shape[2] * weight.shape[3];
  const kernelSize = weight.shape[1] * kernelChannelSize;
  const inputChannelSize = input.shape[2] * input.shape[3];
  const inputSize = input.shape[1] * inputChannelSize;

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
                  // DO NOT use atLoc, it will affect performance
                  const kernelIdx =
                    c * kernelSize +
                    kc * kernelChannelSize +
                    ky * weight.shape[3] +
                    kx;
                  const dataIdx =
                    n * inputSize +
                    kc * inputChannelSize +
                    cy * input.shape[3] +
                    cx;

                  const kernelValue = weight.data[kernelIdx];
                  const dataValue = input.data[dataIdx];

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

  return output;
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
