import { Tensor, TensorBuilder } from "../tensor";

export function handleShape(inputs: Tensor[]): Tensor {
  const shape = inputs[0].shape;
  const output = TensorBuilder.withData(shape);

  return output;
}
