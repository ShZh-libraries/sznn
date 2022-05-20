import init, {
  Tensor,
  initThreadPool,
  handleAbs,
  handleNeg,
  handleSigmoid,
  handleAdd,
  handleMul,
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
} from "../../src/rs/pkg";
import { TensorBuilder as WasmBuilder } from "../../src/tensor";

import * as Comlink from "comlink";
import { tensorList } from "../../src/utils";

// Initialization WebAssembly in dedicated worker
(async () => {
  await init();
  await initThreadPool(navigator.hardwareConcurrency);
})();

// The only way to pass Tensor from main thread to the worker
function wrap(ptr: Tensor) {
  let tensor = new Tensor();
  tensor.free();
  Object.assign(tensor, ptr);

  return tensor;
}

// In order not to lose type information of our tensors
// Use `Comlink.proxy` to wrap the return value
function wrapFn(fn: Function) {
  return function () {
    // Parameters
    var args = [];
    for (const arg of arguments) {
      if (arg instanceof Object && "ptr" in arg) {
        // Tensor obj passed from main thread
        args.push(wrap(arg));
      } else {
        // Other parameter(numbers are most likely)
        args.push(arg);
      }
    }
    // Function call
    const output = fn(...args);
    // Return value
    return Comlink.proxy(output);
  };
}

// Specialization
function handleConcatWrapper(ptrs: Tensor[], axis: number) {
  const tensors = ptrs.map((ptr) => wrap(ptr));
  const list = tensorList(tensors);
  const output = handleConcat(list, axis);
  return Comlink.proxy(output);
}

// Use Comlink to expose the functionality of Web workers
Comlink.expose({
  handleAbs: wrapFn(handleAbs),
  handleNeg: wrapFn(handleNeg),
  handleSigmoid: wrapFn(handleSigmoid),
  handleAdd: wrapFn(handleAdd),
  handleMul: wrapFn(handleMul),
  handlePadding: wrapFn(handlePadding),
  handleConcat: handleConcatWrapper,
  handleConv: wrapFn(handleConv),
  handleMaxPool2D: wrapFn(handleMaxPool2D),
  handleAvgPool2D: wrapFn(handleAvgPool2D),
  handleGlobalAvgPool: wrapFn(handleGlobalAvgPool),
  handleRelu: wrapFn(handleRelu),
  handleLeakyRelu: wrapFn(handleLeakyRelu),
  handleInstanceNorm: wrapFn(handleInstanceNorm),
  handleUpSample: wrapFn(handleUpSample),
  withAllArgs: WasmBuilder.withAllArgs,
});
