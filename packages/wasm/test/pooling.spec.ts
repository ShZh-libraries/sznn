import { expect } from "chai";
import { PoolingAttr } from "../../common/attr/pooling";
import {
  handleMaxPool2D,
  handleAvgPool2D,
  TensorBuilder,
  handleGlobalAvgPool,
} from "./public/init";

describe("Test pooling layer of WASM backend", () => {
  it("Test 2D average pooling layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      ]),
      [1, 2, 3, 3]
    );
    const attr = new PoolingAttr();
    attr.kernelShape = [2, 2];
    attr.strides = [1, 1];

    const output = await handleAvgPool2D(
      input,
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1]
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 2n, 2n, 2n]);
    expect(data).deep.eq([3, 4, 6, 7, 12, 13, 15, 16]);
  });

  it("Test 2D max pooling layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      ]),
      [1, 2, 3, 3]
    );
    const attr = new PoolingAttr();
    attr.kernelShape = [2, 2];
    attr.strides = [1, 1];

    const output = await handleMaxPool2D(
      input,
      attr.kernelShape[0],
      attr.kernelShape[1],
      attr.pads[0],
      attr.pads[1],
      attr.pads[2],
      attr.pads[3],
      attr.strides[0],
      attr.strides[1]
    );

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 2n, 2n, 2n]);
    expect(data).deep.eq([5, 6, 8, 9, 14, 15, 17, 18]);
  });

  it("Test global average pooling layer", async () => {
    const input = await TensorBuilder.withAllArgs(
      new Float32Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      ]),
      [1, 2, 3, 3]
    );

    const output = await handleGlobalAvgPool(input);

    const data = await output.toArray();
    const shape = await output.shapeToArray();

    expect(shape).deep.eq([1n, 2n, 1n, 1n]);
    expect(data).deep.eq([5, 14]);
  });
});
