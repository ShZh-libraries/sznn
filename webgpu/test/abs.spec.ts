import { expect } from 'chai';

import { TensorBuilder } from "../src/tensor";
import { getGPUDevice } from "../src/gpu";
import { handleAbs } from "../src/layers/abs";

describe("Test WebGPU backends for ", function() {
    it("Test abs layer", async function () {
        const device = await getGPUDevice();
        const input = TensorBuilder.withData([-1, 0, 1]);
        const output = await handleAbs(input, device!);

        expect(output.data).deep.equal(new Float32Array([1, 0, 1]));
    });
});
