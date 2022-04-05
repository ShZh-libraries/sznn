import { Tensor, TensorBuilder } from "../tensor";

export function handleRelu(data: Tensor): Tensor {
  let output = TensorBuilder.withShape(data.shape);

  for (let index = 0; index < data.data.length; index++) {
    output.data[index] = data.data[index] > 0 ? data.data[index] : 0;
  }

  return output;
}

export function handleLeakyRelu(data: Tensor, alpha: number): Tensor {
  let output = TensorBuilder.withShape(data.shape);

  for (let index = 0; index < data.data.length; index++) {
    output.data[index] =
      data.data[index] > 0 ? data.data[index] : alpha * data.data[index];
  }

  return output;
}
