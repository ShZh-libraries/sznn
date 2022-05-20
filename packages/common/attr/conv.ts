import { onnx } from "onnx-proto";
import { PaddingAttr } from "./padding";

export class ConvAttr {
  autoPad: string = "NOTSET";
  dilations: number[] = [];
  group: number = 1;
  kernelShape: number[] = [];
  pads: number[] = [0, 0, 0, 0];
  strides: number[] = [1, 1];

  getPaddingAttr(): PaddingAttr {
    let paddingAttr = new PaddingAttr();
    paddingAttr.pads = [
      0,
      0,
      this.pads[0],
      this.pads[1],
      0,
      0,
      this.pads[2],
      this.pads[3],
    ];
    return paddingAttr;
  }
}

export function getConvAttr(attributes: onnx.AttributeProto[]): ConvAttr {
  let result: ConvAttr = new ConvAttr();
  for (const attribute of attributes) {
    switch (attribute.name) {
      case "dilations":
        result.dilations = attribute.ints as number[];
        break;
      case "group":
        result.group = attribute.i as number;
        break;
      case "kernel_shape":
        result.kernelShape = attribute.ints as number[];
        break;
      case "pads":
        result.pads = attribute.ints as number[];
        break;
      case "strides":
        result.strides = attribute.ints as number[];
        break;
      default:
        throw new Error(
          `Unknown attribute ${attribute.name} in Convolutional layer!!`
        );
    }
  }

  return result;
}
