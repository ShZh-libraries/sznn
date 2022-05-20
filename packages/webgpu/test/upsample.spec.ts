import { expect } from "chai";

import { TensorBuilder } from "../src/tensor";
import { getGPUDevice } from "../src/gpu";
import { handleUpSample } from "../src/layers/upsample";

describe("Test upsample layer for GPU backend", () => {
  it("Test normal upsample layer", async () => {
    const device = await getGPUDevice();
    const data = TensorBuilder.withData([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
    const scales = TensorBuilder.withData([1, 1, 2, 3]);
    const result = await handleUpSample(data, scales, device!);

    expect(result.shape).deep.equal([1, 1, 4, 6]);
    expect(result.data).deep.equal(
      new Float32Array([
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
      ])
    );
  });
});
