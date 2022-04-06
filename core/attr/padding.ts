import { onnx } from "onnx-proto";

export class PaddingAttr {
  mode: string = "constant";
  pads: number[] = [];
}

// For ONNX version 2
export function getPaddingAttr(attributes: onnx.AttributeProto[]): PaddingAttr {
  let attr = new PaddingAttr();

  for (let attribute of attributes) {
    switch (attribute.name) {
      case "mode":
        attr.mode = new TextDecoder().decode(attribute.s);
        break;
      case "pads":
        attr.pads = attribute.ints as number[];
        break;
      default:
        throw new Error(`Unrecognize attribute name ${attribute.name}!!`);
    }
  }

  return attr;
}
