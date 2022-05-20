import { expect } from "chai";
import { handleUpSample, TensorBuilder } from "./public/init";

describe("Test upsample layer of WASM backend", () => {
  it("Test normal upsample", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([1, 2, 3, 4]),
      [1, 1, 2, 2]
    );
    const scale = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1, 2, 3]),
      [4]
    );

    const output = await handleUpSample(input, scale);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 4n, 6n]);
    expect(data).deep.eq([
      1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
    ]);
  });
});
