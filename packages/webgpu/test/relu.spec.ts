import { expect } from "chai";

import { TensorBuilder } from "../src/tensor";
import { getGPUDevice } from "../src/gpu";
import { handleRelu, handleLeakyRelu } from "../src/layers/relu";

describe("Test relu layer of WebGPU backend", () => {
  it("Test normal relu layer", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([-1, 0, 1]);
    const output = await handleRelu(input, device!);

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([0, 0, 1]));
  });

  it("Test leaky relu layer", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([-1, 0, 1]);
    const output = await handleLeakyRelu(input, 0.1, device!);

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([0, 0, 0.1]));
  });
});
