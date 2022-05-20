import { ConvAttr } from "../../../common/attr/conv";
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
      for (let out_y = 0; out_y < outputHeight; out_y++) {
        for (let out_x = 0; out_x < outputWidth; out_x++) {
          const y = out_y * attr.strides[0] - attr.pads[0];
          const x = out_x * attr.strides[1] - attr.pads[1];

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
