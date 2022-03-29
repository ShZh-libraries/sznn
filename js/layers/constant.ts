import { onnx } from "onnx-proto";
import { DType, Tensor } from "../tensor";

export function handleConstant(attributes: onnx.AttributeProto[]): Tensor[] {
    let output = new Tensor();
    output.shape = [1];
    output.ndim = 1;

    let buffer = attributes[0].t!.rawData!.buffer.slice(
        attributes[0].t!.rawData!.byteOffset, 
        attributes[0].t!.rawData!.byteOffset + attributes[0].t!.rawData!.byteLength
    );
    switch (attributes[0].t!.dataType) {
        case 1: 
            output.dtype = DType.float32; 
            output.data = new Float32Array(buffer);
            break;
        case 6:
        case 7:     // Currently int64 type is not supported yet
            output.dtype = DType.int32; 
            output.data = new Int32Array(buffer);
            break;
        case 11: 
            output.dtype = DType.float64; 
            output.data = new Float64Array(buffer);
            break;
        default:
            throw Error("Data type not support in ONNX!!");
    }

    return [output];
}