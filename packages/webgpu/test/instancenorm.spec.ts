import { expect } from "chai";

import { TensorBuilder } from "../src";
import { getGPUDevice } from "../src/gpu";
import { handleInstanceNorm } from "../src/layers/instancenorm";

describe("Test instancenorm layer of WebGPU backend", () => {
  it("Test normal instancenorm layer", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([[[[-1, 0, 1]], [[2, 3, 4]]]]);
    const weight = TensorBuilder.withData([1, 1.5]);
    const bias = TensorBuilder.withData([0, 1]);
    const output = await handleInstanceNorm(input, weight, bias, 0, device!);

    expect(output.shape).deep.equal([1, 2, 1, 3]);
    expect(output.data).deep.equal(
      new Float32Array([
        -1.2247449159622192, 0, 1.2247449159622192, -0.8371173739433289, 1,
        2.8371174335479736,
      ])
    );
  });
});
