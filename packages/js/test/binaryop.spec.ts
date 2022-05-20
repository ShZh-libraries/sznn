import { handleBinaryOp } from "../src/layers/binaryop";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backends for binary op", () => {
  test("Test multiplication", () => {
    const a = TensorBuilder.withData([
      [1, 2],
      [3, 4],
    ]);
    const b = TensorBuilder.withData([
      [5, 6],
      [7, 8],
    ]);

    const result = handleBinaryOp(a, b, (x, y) => x * y);
    expect(result.shape).toEqual([2, 2]);
    expect(result.data).toEqual(new Float32Array([5, 12, 21, 32]));
  });

  test("Test broadcast adding", () => {
    const a = TensorBuilder.withData([
      [1, 2],
      [3, 4],
    ]);
    const b = TensorBuilder.withData([2]);

    const result = handleBinaryOp(a, b, (x, y) => x + y);
    expect(result.shape).toEqual([2, 2]);
    expect(result.data).toEqual(new Float32Array([3, 4, 5, 6]));
  });
});
