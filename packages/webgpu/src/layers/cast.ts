import { DType, Tensor } from "../tensor";

// See https://github.com/onnx/onnx/blob/96516aecd4c110b0ac57eba08ac236ebf7205728/onnx/onnx.in.proto#L484
export function handleCast(input: Tensor, to: number): Tensor {
  let output = input.copy();

  switch (to) {
    case 1:
      output.dtype = DType.float32;
      output.data = Float32Array.from(input.data);
      break;
    case 6:
      output.dtype = DType.int32;
      output.data = Int32Array.from(input.data);
      break;
    default:
      throw new Error(`Unsupported to number ${to}!!`);
  }

  return output;
}
