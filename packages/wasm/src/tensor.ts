import { onnx } from "onnx-proto";
import { DType, Tensor } from "./rs/pkg";

export type TensorDataType =
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export class TensorDict {
  private pool: Map<string, Tensor> = new Map();

  get(name: string): Tensor | undefined {
    return this.pool.get(name);
  }

  set(name: string, tensor: Tensor): void {
    this.pool.set(name, tensor);
  }

  init(initializers: onnx.ITensorProto[]) {
    for (const initializer of initializers) {
      const tensor = TensorBuilder.withInitializer(
        initializer as onnx.TensorProto
      );
      this.pool.set((initializer as onnx.TensorProto).name, tensor);
    }

    return this;
  }
}

export class TensorBuilder {
  static withInitializer(initializer: onnx.TensorProto): Tensor {
    let tensor = new Tensor();
    tensor.setShape(initializer.dims as number[]);

    if (initializer.rawData.length != 0) {
      let buffer = initializer.rawData.buffer.slice(
        initializer.rawData.byteOffset,
        initializer.rawData.byteOffset + initializer.rawData.byteLength
      );
      switch (initializer.dataType) {
        case 1:
          tensor.setDataWithF32Array(new Float32Array(buffer));
          break;
        case 6:
        case 7: // Currently int64 type is not supported yet
          tensor.setDataWithI32Array(new Int32Array(buffer));
          break;
        case 11:
          tensor.setDataWithF64Array(new Float64Array(buffer));
          break;
        default:
          throw Error("Data type not support in ONNX!!");
      }
    } else {
      switch (initializer.dataType) {
        case 1:
          tensor.setDataWithF32Array(Float32Array.from(initializer.floatData));
          break;
        case 6:
          tensor.setDataWithI32Array(Int32Array.from(initializer.int32Data));
          break;
        case 7:
          tensor.setDataWithI32Array(
            Int32Array.from(initializer.int64Data as number[])
          );
          break;
        case 11:
          tensor.setDataWithF64Array(Float64Array.from(initializer.doubleData));
          break;
        default:
          throw Error("Data type not support in ONNX!!");
      }
    }

    return tensor;
  }

  static withShape(shape: number[], dtype?: DType): Tensor {
    let tensor = new Tensor();
    tensor.setShape(shape);

    const length = shape.reduceRight((x, y) => x * y);
    const data = [];
    for (let i = 0; i < length; i++) {
      data[i] = 0;
    }
    tensor.setDataWithArray(data, dtype ? dtype : DType.Float32);

    return tensor;
  }

  static withAllArgs(
    data: TensorDataType,
    shape: number[],
    dtype?: DType
  ): Tensor {
    let tensor = new Tensor();

    if (dtype) {
      switch (dtype) {
        case DType.Int8:
          tensor.setDataWithI8Array(data as Int8Array);
          break;
        case DType.Int16:
          tensor.setDataWithI16Array(data as Int16Array);
          break;
        case DType.Int32:
          tensor.setDataWithI32Array(data as Int32Array);
          break;
        case DType.UInt8:
          tensor.setDataWithU8Array(data as Uint8Array);
          break;
        case DType.UInt16:
          tensor.setDataWithU16Array(data as Uint16Array);
          break;
        case DType.UInt32:
          tensor.setDataWithU32Array(data as Uint32Array);
          break;
        case DType.Float32:
          tensor.setDataWithF32Array(data as Float32Array);
          break;
        case DType.Float64:
          tensor.setDataWithF64Array(data as Float64Array);
          break;
      }
    } else {
      tensor.setDataWithF32Array(data as Float32Array);
    }

    tensor.setShape(shape);

    return tensor;
  }
}
