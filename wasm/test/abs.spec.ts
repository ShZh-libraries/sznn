import { expect } from "chai";

import { handleAbs, TensorBuilder, Tensor } from "./public/init";

// Wait for initializaiton done
beforeEach((done) => {
    setTimeout(done, 500);
})

describe("Test abs layer of WASM backend", () => {
    it("Test normal abs layer", async () => {
        const inputPtr = await TensorBuilder.withAllArgs(
            new Float32Array([1, -2, -3, 4]), [2, 2]
        );
        const output: Tensor = await handleAbs(inputPtr);

        const data = await output.toArray();
        const shape = await output.shapeToArray();

        expect(shape).deep.eq([2n, 2n]);
        expect(data).deep.eq([1, 2, 3, 4]);
    })
})