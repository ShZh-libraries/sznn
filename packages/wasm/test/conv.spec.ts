import { expect } from "chai";
import { ConvAttr } from "../../common/attr/conv";
import { handleConv, TensorBuilder } from "./public/init";

describe("Test convolutional layer of WASM backend", () => {
  it("Test simple convolution", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
      ]),
      [1, 1, 5, 5]
    );
    const weight = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
      [1, 1, 3, 3]
    );
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    const output = await handleConv(
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1],
      input,
      weight
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 3n, 3n]);
    expect(data).deep.eq([54, 63, 72, 99, 108, 117, 144, 153, 162]);
  });

  it("Test convolution layer with paddings", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
      ]),
      [1, 1, 5, 5]
    );
    const weight = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
      [1, 1, 3, 3]
    );
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    attr.pads = [1, 1, 1, 1];
    const output = await handleConv(
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1],
      input,
      weight
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 5n, 5n]);
    expect(data).deep.eq([
      12, 21, 27, 33, 24, 33, 54, 63, 72, 51, 63, 99, 108, 117, 81, 93, 144,
      153, 162, 111, 72, 111, 117, 123, 84,
    ]);
  });

  it("Test convolution layer with stride", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
      [1, 1, 7, 5]
    );
    const weight = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
      [1, 1, 3, 3]
    );
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    attr.strides = [2, 2];
    const output = await handleConv(
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1],
      input,
      weight
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 3n, 2n]);
    expect(data).deep.eq([54, 72, 144, 162, 234, 252]);
  });

  it("Test convolution layer with special paddings", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
      [1, 1, 7, 5]
    );
    const weight = await TensorBuilder.withAllArgs(
      new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
      [1, 1, 3, 3]
    );
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    attr.pads = [1, 0, 1, 0];
    attr.strides = [2, 2];

    const output = await handleConv(
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1],
      input,
      weight
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 1n, 4n, 2n]);
    expect(data).deep.eq([21, 33, 99, 117, 189, 207, 171, 183]);
  });
});
