import { PoolingAttr } from "../../common/attr/pooling";
import {
  handleAvgPool2D,
  handleMaxPool2D,
  handleGlobalAvgPool,
} from "../src/layers/pooling";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backend for pooling layer", () => {
  const src = TensorBuilder.withData([
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

  describe("Test avg pooling layer", () => {
    test("Test normal avg pooling layer", () => {
      const avgPoolingAttr = new PoolingAttr();
      avgPoolingAttr.kernelShape = [2, 2];
      avgPoolingAttr.strides = [1, 1];

      const result = handleAvgPool2D(src, avgPoolingAttr);
      expect(result.shape).toEqual([1, 2, 2, 2]);
      expect(result.data).toEqual(
        new Float32Array([3, 4, 6, 7, 12, 13, 15, 16])
      );
    });
  });

  describe("Test max pooling layer", () => {
    test("Test normal max pooling layer", () => {
      const maxPoolingAttr = new PoolingAttr();
      maxPoolingAttr.kernelShape = [2, 2];
      maxPoolingAttr.strides = [1, 1];

      const result = handleMaxPool2D(src, maxPoolingAttr);

      expect(result.shape).toEqual([1, 2, 2, 2]);
      expect(result.data).toEqual(
        new Float32Array([5, 6, 8, 9, 14, 15, 17, 18])
      );
    });
  });

  describe("Test global avg pooling layer", () => {
    test("Test normal global avg pooling layer", () => {
      const results = handleGlobalAvgPool(src);
      expect(results.shape).toEqual([1, 2, 1, 1]);
      expect(results.data).toEqual(new Float32Array([5, 14]));
    });
  });
});
