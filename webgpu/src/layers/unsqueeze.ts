import { Tensor } from "../tensor";

export function handleUnsqueeze(input: Tensor, dims: number[]) {
  let output = input.copy();

  const nonNegativeDims = dims.map((dim) => {
    if (dim < 0) {
      dim += input.ndim;
    }
  });

  let outputShape = [];
  let inputIndex = 0;
  for (let index = 0; index < input.ndim + dims.length; index++) {
    if (dims.includes(index)) {
      outputShape.push(1);
    } else {
      outputShape.push(input.shape[inputIndex++]);
    }
  }

  output.shape = outputShape.slice();

  return output;
}
