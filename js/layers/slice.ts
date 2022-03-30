import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class SliceAttr {
  axes: number[] = [];
  starts: number[] = [];
  ends: number[] = [];
}

function handleAttributes(attributes: onnx.AttributeProto[]): SliceAttr {
  let attr = new SliceAttr();
  for (let attribute of attributes) {
    switch (attribute.name) {
      case "axes":
        attr.axes = attribute.ints as number[];
        break;
      case "starts":
        attr.starts = attribute.ints as number[];
        break;
      case "ends":
        attr.ends = attribute.ints as number[];
        break;
      default:
        throw new Error(`${attribute.name} not supported!!`);
    }
  }

  return attr;
}

export function handleSlice(
  inputs: Tensor[],
  attributes: onnx.AttributeProto[]
): Tensor[] {
  const sliceAttr = handleAttributes(attributes);
  const output = forward(inputs[0], sliceAttr);

  return [output];
}

export function forward(input: Tensor, attr: SliceAttr): Tensor {
  if (attr.axes.length == 1 && attr.axes[0] == 0) {
    const outputShape = attr.ends[0] - attr.starts[0];
    const output = TensorBuilder.withShape([outputShape]);
    for (let i = attr.starts[0]; i < attr.ends[0]; i++) {
      output.data[i - attr.starts[0]] = input.data[i];
    }

    return output;
  }

  return new Tensor();
}
