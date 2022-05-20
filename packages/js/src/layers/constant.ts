import { onnx } from "onnx-proto";
import { DType, Tensor } from "../tensor";

export function handleConstant(attributes: onnx.AttributeProto[]): Tensor {
  let output = new Tensor();
  output.shape = attributes[0].t!.dims! as number[];
  output.ndim = output.shape.length;

  let buffer = attributes[0].t!.rawData!.buffer.slice(
    attributes[0].t!.rawData!.byteOffset,
    attributes[0].t!.rawData!.byteOffset + attributes[0].t!.rawData!.byteLength
  );
  switch (attributes[0].t!.dataType) {
    case 1:
      output.dtype = DType.float32;
      output.data = new Float32Array(buffer);
      break;
    case 2:
      output.dtype = DType.uint8;
      output.data = new Uint8Array(buffer);
      break;
    case 3:
      output.dtype = DType.int8;
      output.data = new Int8Array(buffer);
      break;
    case 4:
      output.dtype = DType.uint16;
      output.data = new Uint16Array(buffer);
      break;
    case 5:
      output.dtype = DType.int16;
      output.data = new Int16Array(buffer);
      break;
    case 6:
      output.dtype = DType.int32;
      output.data = new Int32Array(buffer);
      break;
    case 7: // Currently int64 type is not supported yet
      output.dtype = DType.int32;
      const tempInt64Arr = new BigInt64Array(buffer);
      output.data = new Int32Array([Number(tempInt64Arr[0])]);
      break;
    case 11:
      output.dtype = DType.float64;
      output.data = new Float64Array(buffer);
      break;
    case 12:
      output.dtype = DType.uint32;
      output.data = new Uint32Array(buffer);
      break;
    case 13:
      output.dtype = DType.uint32;
      const tempUInt64Arr = new BigUint64Array(buffer);
      output.data = new Uint32Array([Number(tempUInt64Arr[0])]);
      break;
    default:
      throw Error("Data type not support in ONNX!!");
  }

  return output;
}
