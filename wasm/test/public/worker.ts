import init, { 
    Tensor, initThreadPool, 
    handleAbs, 
    handleNeg,
    handleSigmoid,
    handleAdd,
    handleMul,
    handleConcat,
    handleConv,
    handleMaxPool2D,
    handleAvgPool2D,
    handleGlobalAvgPool,
} from "../../src/rs/pkg";
import { TensorBuilder as WasmBuilder } from "../../src/tensor";

import * as Comlink from "comlink";
import { tensorList } from "../../src/utils";

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
function wrapFn(fn: Function) {
    return function() {
        // Parameters
        var args = [];
        for (let i = 0; i < arguments.length; i++) {
            args.push(wrap(arguments[i]));
        }
        // Function call
        const output = fn(...args);
        // Return value
        return Comlink.proxy(output);
    }
}

// Specialization
function handleConcatWrapper(ptrs: Tensor[], axis: number) {
    const tensors = ptrs.map(ptr => wrap(ptr));
    const list = tensorList(tensors);
    const output = handleConcat(list, axis);
    return Comlink.proxy(output);
}

function handleConvWrapper(kH: number, kW: number, pT: number, pL: number, pB: number, pR: number, sY: number, sX: number, inputPtr: Tensor, weightPtr: Tensor, biasPtr?: Tensor) {
    const input = wrap(inputPtr);
    const weight = wrap(weightPtr);
    const bias = biasPtr? wrap(biasPtr) : biasPtr;

    const output = handleConv(kH, kW, pT, pL, pB, pR, sY, sX, input, weight, bias);

    return Comlink.proxy(output);
}

function handleMaxPoolWrapper(inputPtr: Tensor, kH: number, kW: number, pT: number, pL: number, pB: number, pR: number, sY: number, sX: number) {
    const input = wrap(inputPtr);
    const output = handleMaxPool2D(input, kH, kW, pT, pL, pB, pR, sY, sX);
    return Comlink.proxy(output);
}

function handleAvgPoolWrapper(inputPtr: Tensor, kH: number, kW: number, pT: number, pL: number, pB: number, pR: number, sY: number, sX: number) {
    const input = wrap(inputPtr);
    const output = handleAvgPool2D(input, kH, kW, pT, pL, pB, pR, sY, sX);
    return Comlink.proxy(output);
}


// Use Comlink to expose the functionality of Web workers
Comlink.expose({
    handleAbs: wrapFn(handleAbs),
    handleNeg: wrapFn(handleNeg),
    handleSigmoid: wrapFn(handleSigmoid),
    handleAdd: wrapFn(handleAdd),
    handleMul: wrapFn(handleMul),
    handleConcat: handleConcatWrapper,
    handleConv: handleConvWrapper,
    handleMaxPool2D: handleMaxPoolWrapper,
    handleAvgPool2D: handleAvgPoolWrapper,
    handleGlobalAvgPool: wrapFn(handleGlobalAvgPool),
    withAllArgs: WasmBuilder.withAllArgs,
})


