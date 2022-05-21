import { TensorBuilder } from "../src/tensor";
import { getGPUDevice } from "../src/gpu";
import { handlePadding } from "../src/layers/padding";
import { PaddingAttr } from "../../common/attr/padding";

import { expect } from "chai";

describe("Test padding layer of WebGPU backend", () => {
  it("Test constant padding", async () => {
    const device = await getGPUDevice();
    const input = TensorBuilder.withData([
      [
        [
          [1, 1.2],
          [2.3, 3.4],
          [4.5, 5.7],
        ],
      ],
    ]);
    const attr = new PaddingAttr();
    attr.pads = [0, 0, 0, 2, 0, 0, 0, 0];
    const output = await handlePadding(input, attr, device!);
    expect(output.shape).deep.equal([1, 1, 3, 4]);
    expect(output.data).deep.equal(
      new Float32Array([0, 0, 1, 1.2, 0, 0, 2.3, 3.4, 0, 0, 4.5, 5.7])
    );
  });
});
