import * as Comlink from "comlink";
import { Model as WasmModel } from "./model";
import { DType, Tensor } from "./rs/pkg";
import { TensorDataType } from "./tensor";

const { Model, loadModel, withAllArgs } = Comlink.wrap(
  new Worker(new URL("./worker.ts", import.meta.url))
) as {
  Model: Promise<WasmModel>;
  loadModel: (path: string) => Promise<WasmModel>;
  withAllArgs: (
    data: TensorDataType,
    shape: number[],
    dtype?: DType | undefined
  ) => Promise<Tensor>;
};

class TensorBuilderWrapper {
  static async withAllArgs(
    data: TensorDataType,
    shape: number[],
    dtype?: DType
  ): Promise<Tensor> {
    return await withAllArgs(data, shape, dtype);
  }
}

export { Model, Tensor, loadModel, TensorBuilderWrapper as TensorBuilder };
