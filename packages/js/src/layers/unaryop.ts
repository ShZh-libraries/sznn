import { Tensor, TensorBuilder } from "../tensor";

type UnaryFunc = (x: number) => number;

export function handleUnaryOp(input: Tensor, op: UnaryFunc): Tensor {
  let output = TensorBuilder.withShape(input.shape);

  for (let index = 0; index < input.data.length; index++) {
    output.data[index] = op(input.data[index]);
  }

  return output;
}
