import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export function handleRelu(inputs: Tensor[]): Tensor[] {
  const data = inputs[0];
  let output = forwardRelu(data);

  return [output];
}

export function handleLeakyRelu(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const data = inputs[0];
  const alpha = attributes[0].f;
  let output = forwardLeakRelu(data, alpha);

  return [output];
}

export function forwardRelu(data: Tensor): Tensor {
  let output = TensorBuilder.withShape(data.shape);

  for (let index = 0; index < data.data.length; index++) {
    output.data[index] = data.data[index] > 0 ? data.data[index] : 0;
  }

  return output;
}

export function forwardLeakRelu(data: Tensor, alpha: number): Tensor {
  let output = TensorBuilder.withShape(data.shape);

  for (let index = 0; index < data.data.length; index++) {
    output.data[index] =
      data.data[index] > 0 ? data.data[index] : alpha * data.data[index];
  }

  return output;
}
