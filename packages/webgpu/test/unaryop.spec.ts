import { expect } from "chai";

import { TensorBuilder } from "../src/tensor";
import { getGPUDevice } from "../src/gpu";
import { handleUnaryOp } from "../src/layers/unaryop";

describe("Test unaryop layer of WebGPU backend", () => {
  it("Test identity operation", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([-1, 0, 1]);
    const output = await handleUnaryOp(
      input,
      "output[global_id.x] = input[global_id.x]",
      device!
    );

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([-1, 0, 1]));
  });

  it("Test abs operation", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([-1, 0, 1]);
    const output = await handleUnaryOp(
      input,
      "output[global_id.x] = abs(input[global_id.x])",
      device!
    );

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([1, 0, 1]));
  });

  it("Test negative operation", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([1, 0, -1]);
    const output = await handleUnaryOp(
      input,
      "output[global_id.x] = -input[global_id.x]",
      device!
    );

    expect(output.shape).deep.equal([3]);
    expect(output.data).deep.equal(new Float32Array([-1, -0, 1]));
  });
});
