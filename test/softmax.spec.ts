import { forward } from "../js/layers/softmax";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backend for softmax layer", () => {
  test("Test default softmax", () => {
    const src = TensorBuilder.withData([[-1, 0, 1]]);
    const result = forward(src);

    expect(result.shape).toEqual([1, 3]);
    expect(result.data).toEqual(
      new Float32Array([0.09003057, 0.24472848, 0.66524094])
    );
  });
});
