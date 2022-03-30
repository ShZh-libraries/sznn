import { onnx } from "onnx-proto";
import { Tensor } from "../tensor";

export function handleInstanceNorm(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const epsilon = attributes[0].f;
  const output = forward(inputs[0], inputs[1], inputs[2], epsilon);

  return [output];
}

// Not consider number dim
export function forward(
  input: Tensor,
  weight: Tensor,
  bias: Tensor,
  epsilon: number
): Tensor {
  let output = input.copy();
  const channelSize = output.shape[2] * output.shape[3];

  for (let c = 0; c < output.shape[1]; c++) {
    // Calculate mean
    let mean = 0;
    for (let i = 0; i < channelSize; i++) {
      mean += input.data[c * channelSize + i];
    }
    mean /= channelSize;
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < channelSize; i++) {
      let stdVar = input.data[c * channelSize + i] - mean;
      variance += stdVar * stdVar;
    }
    variance /= channelSize;
    // Normalization
    for (let i = 0; i < channelSize; i++) {
      output.data[c * channelSize + i] =
        (weight.data[c] * (input.data[c * channelSize + i] - mean)) /
          Math.sqrt(variance + epsilon) +
        bias.data[c];
    }
  }

  return output;
}
