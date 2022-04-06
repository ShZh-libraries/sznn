import { Tensor, TensorBuilder } from "../tensor";

// For 1 x C x W x H tensor:
export function handleSoftmax(input: Tensor): Tensor {
  const output = TensorBuilder.withShape(input.shape);

  let max = input.data[0];
  for (let c = 0; c < input.shape[1]; c++) {
    if (max < input.data[c]) {
      max = input.data[c];
    }
  }

  let sum = 0;
  for (let c = 0; c < input.shape[1]; c++) {
    // Partial inplace
    input.data[c] = Math.exp(input.data[c] - max);
    sum += input.data[c];
  }

  for (let c = 0; c < input.shape[1]; c++) {
    output.data[c] = input.data[c] / sum;
  }

  return output;
}
