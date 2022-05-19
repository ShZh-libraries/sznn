import { TensorBuilder } from "../src/tensor";
import { handleConv } from "../src/layers/conv";
import { ConvAttr } from "../../common/attr/conv";

describe("Test JS backend for convolutional layer", () => {
  describe("Test convolution with/without paddings", () => {
    const data = TensorBuilder.withData([
      [
        [
          [0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [20, 21, 22, 23, 24],
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

    test("Test simple convolution", () => {
      const attr = new ConvAttr();
      attr.kernelShape = [3, 3];

      const result = handleConv(attr, data, weight);

      expect(result.shape).toEqual([1, 1, 3, 3]);
      expect(result.data).toEqual(
        new Float32Array([54, 63, 72, 99, 108, 117, 144, 153, 162])
      );
    });
    test("Test convolution with paddings", () => {
      const attr = new ConvAttr();
      attr.kernelShape = [3, 3];
      attr.pads = [1, 1, 1, 1];

      const result = handleConv(attr, data, weight);

      expect(result.shape).toEqual([1, 1, 5, 5]);
      expect(result.data).toEqual(
        new Float32Array([
          12, 21, 27, 33, 24, 33, 54, 63, 72, 51, 63, 99, 108, 117, 81, 93, 144,
          153, 162, 111, 72, 111, 117, 123, 84,
        ])
      );
    });
  });

  describe("Test convolution with stride", () => {
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

    test("Test convolution with stride", () => {
      const attr = new ConvAttr();
      attr.kernelShape = [3, 3];
      attr.strides = [2, 2];

      const result = handleConv(attr, data, weight);

      expect(result.shape).toEqual([1, 1, 3, 2]);
      expect(result.data).toEqual(
        new Float32Array([54, 72, 144, 162, 234, 252])
      );
    });

    test("Test convolution with special paddings", () => {
      const attr = new ConvAttr();
      attr.kernelShape = [3, 3];
      attr.pads = [1, 0, 1, 0];
      attr.strides = [2, 2];

      const result = handleConv(attr, data, weight);

      expect(result.shape).toEqual([1, 1, 4, 2]);
      expect(result.data).toEqual(
        new Float32Array([21, 33, 99, 117, 189, 207, 171, 183])
      );
    });
  });
});
