import { handleConcat } from "../src/layers/concat";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backends for concat layer", () => {
  test("Test 1D tesnor concat", () => {
    const src1 = TensorBuilder.withData([1, 2, 3]);
    const src2 = TensorBuilder.withData([4, 5, 6]);

    const result = handleConcat([src1, src2], 0);
    expect(result.shape).toEqual([6]);
    expect(result.data).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  describe("Test 2D tensor concat", () => {
    const src1 = TensorBuilder.withData([
      [1, 1, 1],
      [2, 2, 2],
    ]);
    const src2 = TensorBuilder.withData([
      [3, 3, 3],
      [4, 4, 4],
    ]);

    test("Test 2D tensor concat with axis0", () => {
      const result = handleConcat([src1, src2], 0);
      expect(result.shape).toEqual([4, 3]);
      expect(result.data).toEqual(
        new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
      );
    });

    test("Test 2D tensor concat with axis1", () => {
      const result = handleConcat([src1, src2], 1);
      expect(result.shape).toEqual([2, 6]);
      expect(result.data).toEqual(
        new Float32Array([1, 1, 1, 3, 3, 3, 2, 2, 2, 4, 4, 4])
      );
    });
  });

  describe("Test 3D tensor concat", () => {
    const src1 = TensorBuilder.withData([
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
      [
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27],
      ],
    ]);
    const src2 = TensorBuilder.withData([
      [
        [-1, -2, -3],
        [-4, -5, -6],
        [-7, -8, -9],
      ],
      [
        [-10, -11, -12],
        [-13, -14, -15],
        [-16, -17, -18],
      ],
      [
        [-19, -20, -21],
        [-22, -23, -24],
        [-25, -26, -27],
      ],
    ]);

    test("Test 3D tensor concat with axis0", () => {
      const result = handleConcat([src1, src2], 0);
      expect(result.shape).toEqual([6, 3, 3]);
      expect(result.data).toEqual(
        new Float32Array([
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,
          -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24,
          -25, -26, -27,
        ])
      );
    });

    test("Test 3D tensor concat with axis1", () => {
      const result = handleConcat([src1, src2], 1);
      expect(result.shape).toEqual([3, 6, 3]);
      expect(result.data).toEqual(
        new Float32Array([
          1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, 10, 11,
          12, 13, 14, 15, 16, 17, 18, -10, -11, -12, -13, -14, -15, -16, -17,
          -18, 19, 20, 21, 22, 23, 24, 25, 26, 27, -19, -20, -21, -22, -23, -24,
          -25, -26, -27,
        ])
      );
    });

    test("Test 3D tensor concat with axis2", () => {
      const result = handleConcat([src1, src2], 2);
      expect(result.shape).toEqual([3, 3, 6]);
      expect(result.data).toEqual(
        new Float32Array([
          1, 2, 3, -1, -2, -3, 4, 5, 6, -4, -5, -6, 7, 8, 9, -7, -8, -9, 10, 11,
          12, -10, -11, -12, 13, 14, 15, -13, -14, -15, 16, 17, 18, -16, -17,
          -18, 19, 20, 21, -19, -20, -21, 22, 23, 24, -22, -23, -24, 25, 26, 27,
          -25, -26, -27,
        ])
      );
    });
  });
});
