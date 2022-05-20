import { expect } from "chai";

import {
  handleAbs,
  handleNeg,
  handleSigmoid,
  TensorBuilder,
} from "./public/init";

describe("Test unary op of WASM backend", () => {
  it("Test normal abs layer", async () => {
    const inputPtr = await TensorBuilder.withAllArgs(
      new Float32Array([1, -2, -3, 4]),
      [2, 2]
    );
    const output = await handleAbs(inputPtr);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([2n, 2n]);
    expect(data).deep.eq([1, 2, 3, 4]);
  });

  it("Test negative layer", async () => {
    const inputPtr = await TensorBuilder.withAllArgs(
      new Float32Array([1, -2, -3, 4]),
      [2, 2]
    );
    const output = await handleNeg(inputPtr);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([2n, 2n]);
    expect(data).deep.eq([-1, 2, 3, -4]);
  });

  it("Test sigmoid layer", async () => {
    const inputPtr = await TensorBuilder.withAllArgs(
      new Float32Array([1, -1, 0]),
      [3]
    );
    const output = await handleSigmoid(inputPtr);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([0.2689414322376251, 0.7310585975646973, 0.5]);
  });
});
