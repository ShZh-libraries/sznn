import { expect } from "chai";
import { ConvAttr } from "../../common/attr/conv";
import { getGPUDevice } from "../src/gpu";
import { handleConv } from "../src/layers/conv";
import { TensorBuilder } from "../src/tensor";

describe("Test convolutional layer of WebGPU backend", () => {
  const data = TensorBuilder.withData([
    [
      [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34],
      ],
    ],
  ]);
  const weight = TensorBuilder.withData([
    [
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
    ],
  ]);

  it("Test convolutional layer with stride", async () => {
    const device = await getGPUDevice();
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    attr.strides = [2, 2];

    const output = await handleConv(device!, attr, data, weight);

    expect(output.shape).deep.equal([1, 1, 3, 2]);
    expect(output.data).deep.equal(
      new Float32Array([54, 72, 144, 162, 234, 252])
    );
  });

  it("Test convolutional layer with padding", async () => {
    const device = await getGPUDevice();
    const attr = new ConvAttr();
    attr.kernelShape = [3, 3];
    attr.pads = [1, 0, 1, 0];
    attr.strides = [2, 2];

    const output = await handleConv(device!, attr, data, weight);
    expect(output.shape).deep.equal([1, 1, 4, 2]);
    expect(output.data).deep.equal(
      new Float32Array([21, 33, 99, 117, 189, 207, 171, 183])
    );
  });
});
