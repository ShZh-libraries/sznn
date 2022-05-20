import { expect } from "chai";
import { TensorBuilder, handleAdd, handleMul } from "./public/init";

describe("Test binary op of WASM backends", () => {
  it("Test add operation", async () => {
    const a = await TensorBuilder.withAllArgs(new Float32Array([1, 2, 3]), [3]);
    const b = await TensorBuilder.withAllArgs(new Float32Array([4, 5, 6]), [3]);
    const output = await handleAdd(a, b);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([5, 7, 9]);
  });

  it("Test add operation with broadcast", async () => {
    const a = await TensorBuilder.withAllArgs(new Float32Array([1, 2, 3]), [3]);
    const b = await TensorBuilder.withAllArgs(new Float32Array([1]), [1]);
    const output = await handleAdd(a, b);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([2, 3, 4]);
  });

  it("Test multiply operation", async () => {
    const a = await TensorBuilder.withAllArgs(new Float32Array([1, 2, 3]), [3]);
    const b = await TensorBuilder.withAllArgs(new Float32Array([2]), [1]);
    const output = await handleMul(a, b);

    const data = await output.toArray();
    const shape = await output.shapeToArray();
    expect(shape).deep.eq([3n]);
    expect(data).deep.eq([2, 4, 6]);
  });
});
