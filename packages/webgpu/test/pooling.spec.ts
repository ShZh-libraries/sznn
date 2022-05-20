import { expect } from "chai";
import { PoolingAttr } from "../../common/attr/pooling";
import { getGPUDevice } from "../src/gpu";
import {
  handleAvgPool2D,
  handleGlobalAvgPool,
  handleMaxPool2D,
} from "../src/layers/pooling";
import { TensorBuilder } from "../src/tensor";

describe("Test pooling layer of WebGPU backend", () => {
  const data = TensorBuilder.withData([
    [
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
      ],
    ],
  ]);

  it("Test average pooling layer", async () => {
    const device = await getGPUDevice();
    const attr = new PoolingAttr();
    attr.kernelShape = [2, 2];
    attr.strides = [1, 1];

    const output = await handleAvgPool2D(data, attr, device!);
    expect(output.shape).deep.equal([1, 2, 2, 2]);
    expect(output.data).deep.equal(
      new Float32Array([3, 4, 6, 7, 12, 13, 15, 16])
    );
  });

  it("Test max pooling layer", async () => {
    const device = await getGPUDevice();
    const attr = new PoolingAttr();
    attr.kernelShape = [2, 2];
    attr.strides = [1, 1];

    const output = await handleMaxPool2D(data, attr, device!);
    expect(output.shape).deep.equal([1, 2, 2, 2]);
    expect(output.data).deep.equal(
      new Float32Array([5, 6, 8, 9, 14, 15, 17, 18])
    );
  });

  it("Test global average  pooling layer", async () => {
    const device = await getGPUDevice();
    const output = await handleGlobalAvgPool(data, device!);
    expect(output.shape).deep.equal([1, 2, 1, 1]);
    expect(output.data).deep.equal(new Float32Array([5, 14]));
  });
});
