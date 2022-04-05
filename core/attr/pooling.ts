import { onnx } from "onnx-proto";

export class PoolingAttr {
    kernelShape: number[] = [];
    pads: number[] = [0, 0, 0, 0];
    strides: number[] = [1, 1];
}

export function getPoolingAttr(attributes: onnx.AttributeProto[]): PoolingAttr {
    let result: PoolingAttr = new PoolingAttr();
    for (const attribute of attributes) {
        switch (attribute.name) {
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
                    `Unknown attribute ${attribute.name} in MaxPooling layer!!`
                );
        }
    }

    return result;
}