import * as Comlink from "comlink";
import { DType, Tensor } from "../../src/rs/pkg";
import { TensorDataType } from "../../src/tensor";

// Import functionality from Web workers
// And re-export at the end
const {
  handleAbs,
  handleNeg,
  handleSigmoid,
  handleAdd,
  handleMul,
  handlePadding,
  handleConcat,
  handleConv,
  handleMaxPool2D,
  handleAvgPool2D,
  handleGlobalAvgPool,
  handleRelu,
  handleLeakyRelu,
  handleInstanceNorm,
  handleUpSample,
  withAllArgs,
} = Comlink.wrap(new Worker(new URL("./worker.ts", import.meta.url))) as {
  handleAbs: (input: Tensor) => Promise<Tensor>;
  handleNeg: (input: Tensor) => Promise<Tensor>;
  handleSigmoid: (input: Tensor) => Promise<Tensor>;
  handleAdd: (a: Tensor, b: Tensor) => Promise<Tensor>;
  handleMul: (a: Tensor, b: Tensor) => Promise<Tensor>;
  handlePadding: (
    input: Tensor,
    pT: number,
    pL: number,
    pB: number,
    pR: number
  ) => Promise<Tensor>;
  handleConcat: (inputs: Tensor[], axis: number) => Promise<Tensor>;
  handleConv: (
    kH: number,
    kW: number,
    pT: number,
    pL: number,
    pB: number,
    pR: number,
    sY: number,
    sX: number,
    input: Tensor,
    weight: Tensor,
    bias?: Tensor
  ) => Promise<Tensor>;
  handleMaxPool2D: (
    input: Tensor,
    kH: number,
    kW: number,
    pT: number,
    pL: number,
    pB: number,
    pR: number,
    sY: number,
    sX: number
  ) => Promise<Tensor>;
  handleAvgPool2D: (
    input: Tensor,
    kH: number,
    kW: number,
    pT: number,
    pL: number,
    pB: number,
    pR: number,
    sY: number,
    sX: number
  ) => Promise<Tensor>;
  handleGlobalAvgPool: (input: Tensor) => Promise<Tensor>;
  handleRelu: (input: Tensor) => Promise<Tensor>;
  handleLeakyRelu: (input: Tensor, alpha: number) => Promise<Tensor>;
  handleInstanceNorm: (
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: number
  ) => Promise<Tensor>;
  handleUpSample: (input: Tensor, scale: Tensor) => Promise<Tensor>;
  withAllArgs: (
    data: TensorDataType,
    shape: number[],
    dtype?: DType | undefined
  ) => Promise<Tensor>;
};

// We cannot use a class with its static method in Comlink
// But we can export functions in Web workers and wrap it as static method in main thread
class TensorBuilderWrapper {
  static async withAllArgs(
    data: TensorDataType,
    shape: number[],
    dtype?: DType
  ): Promise<Tensor> {
    return await withAllArgs(data, shape, dtype);
  }
}

export {
  Tensor,
  handleAbs,
  handleMul,
  handleNeg,
  handleSigmoid,
  handleAdd,
  handleConcat,
  handleConv,
  handlePadding,
  handleMaxPool2D,
  handleAvgPool2D,
  handleGlobalAvgPool,
  handleRelu,
  handleLeakyRelu,
  handleInstanceNorm,
  handleUpSample,
  TensorBuilderWrapper as TensorBuilder,
};
