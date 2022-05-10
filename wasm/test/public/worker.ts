import init, { initThreadPool, handleAbs as handleAbsWasm, Tensor } from "../../src/rs/pkg";
import { TensorBuilder as WasmBuilder } from "../../src/tensor";

import * as Comlink from "comlink";

// Initialization WebAssembly in dedicated worker
(async () => {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
})()

// The only way to pass Tensor from main thread to the worker
function wrap(ptr: Tensor) {
    let tensor = new Tensor();
    tensor.free();
    Object.assign(tensor, ptr);

    return tensor;
}

// In order not to lose type information of our tensors
// Use `Comlink.proxy` to wrap the return value

function handleAbs(ptr: Tensor) {
    const input = wrap(ptr);
    const output = handleAbsWasm(input);

    return Comlink.proxy(output);
}

// Use Comlink to expose the functionality of Web workers
Comlink.expose({
    handleAbs,
    withAllArgs: WasmBuilder.withAllArgs,
})


