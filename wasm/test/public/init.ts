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
    handleConcat,
    withAllArgs 
} = Comlink.wrap(
    new Worker(new URL("./worker.ts", import.meta.url))
) as {
    handleAbs: (input: Tensor) => Promise<Tensor>;
    handleNeg: (input: Tensor) => Promise<Tensor>;
    handleSigmoid: (input: Tensor) => Promise<Tensor>;
    handleAdd: (a: Tensor, b: Tensor) => Promise<Tensor>;
    handleMul: (a: Tensor, b: Tensor) => Promise<Tensor>;
    handleConcat: (inputs: Tensor[], axis: number) => Promise<Tensor>;
    withAllArgs: (data: TensorDataType, shape: number[], dtype?: DType | undefined) => Promise<Tensor>;
}

// We cannot use a class with its static method in Comlink
// But we can export functions in Web workers and wrap it as static method in main thread
class TensorBuilderWrapper {
    static async withAllArgs(data: TensorDataType, shape: number[], dtype?: DType): Promise<Tensor> {
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
    TensorBuilderWrapper as TensorBuilder
};