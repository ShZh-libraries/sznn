import { Tensor, TensorBuilder } from "../tensor";

// TODO: attribute like epsilon
export function handleBatchNorm(
  data: Tensor,
  scale: Tensor,
  bias: Tensor,
  mean: Tensor,
  variance: Tensor
): Tensor {
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

  return output;
}
