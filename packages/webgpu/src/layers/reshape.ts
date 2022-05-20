import { Tensor, TensorBuilder } from "../tensor";

// TODO: forward inplace!
export function handleReshape(input: Tensor, shape: number[]): Tensor {
  // Deal with zero
  for (let index = 0; index < shape.length; index++) {
    if (shape[index] == 0) {
      shape[index] = input.shape[index];
    }
  }

  // Deal with negative numbers
  const negativeIndex = shape.findIndex((x) => x == -1);
  if (negativeIndex != -1) {
    const remainedLength = shape
      .filter((x) => x != -1)
      .reduceRight((x, y) => x * y);
    const remainedSize = input.data.length / remainedLength;
    shape[negativeIndex] = remainedSize;
  }

  const output = TensorBuilder.withAllArgs(input.data, shape, input.dtype);
  return output;
}
