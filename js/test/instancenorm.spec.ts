import { handleInstanceNorm } from "../src/layers/instancenorm";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backends for instance normalization", () => {
  test("Test instance normalization", () => {
    const input = TensorBuilder.withData([[[[-1, 0, 1]], [[2, 3, 4]]]]);
    const weight = TensorBuilder.withData([1, 1.5]);
    const bias = TensorBuilder.withData([0, 1]);

    const result = handleInstanceNorm(input, weight, bias, 0);
    expect(result.shape).toEqual([1, 2, 1, 3]);
    expect(result.data).toEqual(
      new Float32Array([
        -1.224744871391589, 0, 1.224744871391589, -0.8371173070873834, 1,
        2.8371173070873834,
      ])
    );
  });
});
