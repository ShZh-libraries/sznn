import { onnx } from "onnx-proto";
import { Tensor } from "./rs/pkg";

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

class TensorBuilder {
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
}
