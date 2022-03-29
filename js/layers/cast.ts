import { onnx } from "onnx-proto";
import { DType, Tensor } from "../tensor";

export function handleCast(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    let to = attributes[0].i as number;
    const output = forawrd(inputs[0], to);

    return [output];
}

// See https://github.com/onnx/onnx/blob/96516aecd4c110b0ac57eba08ac236ebf7205728/onnx/onnx.in.proto#L484
export function forawrd(input: Tensor, to: number): Tensor {
    let output = input.copy();

    switch (to) {
        case 1:
            output.dtype = DType.float32;
            output.data = Float32Array.from(input.data);
            break;
        case 2:
            output.dtype = DType.uint8;
            output.data = Uint8Array.from(input.data);
            break;
        case 3:
            output.dtype = DType.int8;
            output.data = Int8Array.from(input.data);
            break;
        case 4:
            output.dtype = DType.uint16;
            output.data = Uint16Array.from(input.data);
            break;
        case 5:
            output.dtype = DType.int16;
            output.data = Int16Array.from(input.data);
            break;
        case 6:
            output.dtype = DType.int32;
            output.data = Int32Array.from(input.data);
            break;
        case 11:
            output.dtype = DType.float64;
            output.data = Float64Array.from(input.data);
            break;
        default:
            throw new Error(`Unsupported to number ${to}!!`);
    }

    return output;
}