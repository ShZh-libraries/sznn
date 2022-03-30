import { Tensor, TensorBuilder } from "../tensor";

// TODO: attribute like epsilon
export function handleBatchNorm(inputs: Tensor[]): Tensor[] {
  // Extract weights
  const data = inputs[0];
  const scale = inputs[1];
  const bias = inputs[2];
  const mean = inputs[3];
  const variance = inputs[4];

  // Calculate shape
  let output = TensorBuilder.withShape(data.shape);

  const dataLength = data.data.length;
  for (let index = 0; index < dataLength; index++) {
    output.data[index] =
      ((data.data[index] - mean.data[index]) /
        Math.sqrt(variance.data[index])) *
        scale.data[index] +
      bias.data[index];
  }

  return [output];
}
