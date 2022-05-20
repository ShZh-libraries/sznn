import { getGPUDevice } from "../src/gpu";
import { TensorBuilder } from "../src/tensor";
import { handleBinaryOp } from "../src/layers/binaryop";

import { expect } from "chai";

describe("Test binaryop layer of WebGPU backend", () => {
  it("Test add operation without broadcast", async () => {
    const device = await getGPUDevice();
    const a = TensorBuilder.withData([1, 2, 3]);
    const b = TensorBuilder.withData([4, 5, 6]);
    const output = await handleBinaryOp(a, b, "+", device!);

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([5, 7, 9]));
  });

  it("Test multi-dimension sub operation without broadcast", async () => {
    const device = await getGPUDevice();
    const a = TensorBuilder.withData([1, 2, 3]);
    const b = TensorBuilder.withData([2]);
    const output = await handleBinaryOp(a, b, "-", device!);

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([-1, 0, 1]));
  });

  it("Test multiple operation with broadcast", async () => {
    const device = await getGPUDevice();
    const a = TensorBuilder.withData([1, 2, 3]);
    const b = TensorBuilder.withData([2]);
    const output = await handleBinaryOp(a, b, "*", device!);

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([2, 4, 6]));
  });
});
