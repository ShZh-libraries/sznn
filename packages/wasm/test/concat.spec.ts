import { expect } from "chai";
import { handleConcat, TensorBuilder } from "./public/init";

describe("Test concat layer of WASM backend", () => {
  it("Test concat 1D", async () => {
    const input1 = await TensorBuilder.withAllArgs(new Int8Array([1, 2, 3]), [
      3,
    ]);
    const input2 = await TensorBuilder.withAllArgs(new Int8Array([4, 5, 6]), [
      3,
    ]);

    const output = await handleConcat([input1, input2], 0);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([6n]);
    expect(data).deep.eq([1, 2, 3, 4, 5, 6]);
  });

  it("Test concat 2D with axis 0", async () => {
    const input1 = await TensorBuilder.withAllArgs(
      new Int8Array([1, 1, 1, 2, 2, 2]),
      [2, 3]
    );
    const input2 = await TensorBuilder.withAllArgs(
      new Int8Array([3, 3, 3, 4, 4, 4]),
      [2, 3]
    );

    const output = await handleConcat([input1, input2], 0);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([4n, 3n]);
    expect(data).deep.eq([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
  });

  it("Test concat 2D with axis 1", async () => {
    const input1 = await TensorBuilder.withAllArgs(
      new Int8Array([1, 1, 1, 2, 2, 2]),
      [2, 3]
    );
    const input2 = await TensorBuilder.withAllArgs(
      new Int8Array([3, 3, 3, 4, 4, 4]),
      [2, 3]
    );

    const output = await handleConcat([input1, input2], 1);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([4n, 3n]);
    expect(data).deep.eq([1, 1, 1, 3, 3, 3, 2, 2, 2, 4, 4, 4]);
  });

  it("Test concat 3D with axis 0", async () => {
    const input1 = await TensorBuilder.withAllArgs(
      new Int8Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
      ]),
      [3, 3, 3]
    );
    const input2 = await TensorBuilder.withAllArgs(
      new Int8Array([
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27,
      ]),
      [3, 3, 3]
    );

    const output = await handleConcat([input1, input2], 0);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([6n, 3n, 3n]);
    expect(data).deep.eq([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
      22, 23, 24, 25, 26, 27, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
      -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27,
    ]);
  });

  it("Test concat 3D with axis 1", async () => {
    const input1 = await TensorBuilder.withAllArgs(
      new Int8Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
      ]),
      [3, 3, 3]
    );
    const input2 = await TensorBuilder.withAllArgs(
      new Int8Array([
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27,
      ]),
      [3, 3, 3]
    );

    const output = await handleConcat([input1, input2], 1);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([3n, 6n, 3n]);
    expect(data).deep.eq([
      1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, -10, -11, -12, -13, -14, -15, -16, -17, -18, 19,
      20, 21, 22, 23, 24, 25, 26, 27, -19, -20, -21, -22, -23, -24, -25, -26,
      -27,
    ]);
  });

  it("Test concat 3D with axis 2", async () => {
    const input1 = await TensorBuilder.withAllArgs(
      new Int8Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
      ]),
      [3, 3, 3]
    );
    const input2 = await TensorBuilder.withAllArgs(
      new Int8Array([
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27,
      ]),
      [3, 3, 3]
    );

    const output = await handleConcat([input1, input2], 2);

    const shape = await output.shapeToArray();
    const data = await output.toArray();

    expect(shape).deep.eq([3n, 3n, 6n]);
    expect(data).deep.eq([
      1, 2, 3, -1, -2, -3, 4, 5, 6, -4, -5, -6, 7, 8, 9, -7, -8, -9, 10, 11, 12,
      -10, -11, -12, 13, 14, 15, -13, -14, -15, 16, 17, 18, -16, -17, -18, 19,
      20, 21, -19, -20, -21, 22, 23, 24, -22, -23, -24, 25, 26, 27, -25, -26,
      -27,
    ]);
  });
});
