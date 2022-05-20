import { expect } from "chai";
import { handleLeakyRelu, handleRelu, TensorBuilder } from "./public/init";

describe("Test relu layer of WASM backend", () => {
  it("Test normal relu layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([-1, 0, 1]),
      [3]
    );
    const output = await handleRelu(input);

    const data = await output.toArray();
    const shape = await output.shapeToArray();
    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([0, 0, 1]);
  });

  it("Test leaky relu layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([-1, 0, 1]),
      [3]
    );
    const output = await handleLeakyRelu(input, 0.5);

    const data = await output.toArray();
    const shape = await output.shapeToArray();
    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([-0.5, 0, 1]);
  });
});
