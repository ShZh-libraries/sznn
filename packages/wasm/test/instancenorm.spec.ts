import { expect } from "./index";
import { handleInstanceNorm, TensorBuilder } from "./public/init";

describe("Test instancenorm layer of WASM backend", () => {
  it("Test normal instancenorm layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([-1, 0, 1, 2, 3, 4]),
      [1, 2, 1, 3]
    );
    const weight = await TensorBuilder.withAllArgs(new Float32Array([1, 1.5]), [
      2,
    ]);
    const bias = await TensorBuilder.withAllArgs(new Float32Array([0, 1]), [2]);

    const output = await handleInstanceNorm(input, weight, bias, 0);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 2n, 1n, 3n]);
    expect(data).deep.closeTo(
      [
        -1.2247449159622192, 0, 1.2247449159622192, -0.8371173739433289, 1,
        2.8371174335479736,
      ],
      1e-5
    );
  });
});
