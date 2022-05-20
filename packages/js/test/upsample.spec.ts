import { handleUpSample } from "../src/layers/upsample";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backends for upsample layer", () => {
  test("Test normal upsample layer", () => {
    const data = TensorBuilder.withData([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
    const scales = TensorBuilder.withData([1, 1, 2, 3]);

    const result = handleUpSample(data, scales);
    expect(result.shape).toEqual([1, 1, 4, 6]);

    expect(result.data).toEqual(
      new Float32Array([
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
      ])
    );
  });
});
