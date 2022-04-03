import { onnx } from "onnx-proto";
import { Tensor } from "./rs/pkg";

export function handle(
    opType: string,
    inputs: Tensor[],
    attrs: onnx.AttributeProto[]
  ): Tensor[] {
      return [new Tensor()]
  }