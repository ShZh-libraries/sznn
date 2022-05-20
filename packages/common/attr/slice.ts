import { onnx } from "onnx-proto";

export class SliceAttr {
  axes: number[] = [];
  starts: number[] = [];
  ends: number[] = [];
}

export function getSliceAttr(attributes: onnx.AttributeProto[]): SliceAttr {
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
