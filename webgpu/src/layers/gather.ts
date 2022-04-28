import { Tensor, TensorBuilder } from "../tensor";

// With axis = 0 and inputs[2] only has 1 dim
export function handleGather(input: Tensor, indices: Tensor): Tensor {
  // let outputShape = input.shape.slice();
  // outputShape[indices.data[0]] = 1;

  let output = TensorBuilder.withShape([1]);
  output.data[0] = input.data[indices.data[0]];

  // let outputLength = outputShape.reduceRight((x, y) => x * y);
  // const channelSize = input.shape[2] * input.shape[3];
  // for (let index = 0; index < outputLength; index++) {
  //     output.data[index] = input.data[channelSize * indices.data[0] + index];
  // }

  return output;
}
