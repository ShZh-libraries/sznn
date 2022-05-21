import { expect } from "./index";
import { handlePadding, TensorBuilder } from "./public/init";

describe("Test padding layer of WASM backend", () => {
  it("Test constant padding", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1.2, 2.3, 3.4, 4.5, 5.7]),
      [1, 1, 3, 2]
    );
    const output = await handlePadding(input, 0, 2, 0, 0);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 3n, 4n]);
    expect(data).to.be.deep.closeTo(
      [0, 0, 1, 1.2, 0, 0, 2.3, 3.4, 0, 0, 4.5, 5.7],
      1e-4
    );
  });
});
