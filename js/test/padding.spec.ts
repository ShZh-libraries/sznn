import { PaddingAttr } from "../../common/attr/padding";
import { handlePadding } from "../src/layers/padding";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backend for padding layer", () => {
  const input = TensorBuilder.withData([
    [1, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
  ]);
  const attr = new PaddingAttr();
  attr.pads = [0, 2, 0, 0];

  test("Test constant padding", () => {
    attr.mode = "constant";

    const output = handlePadding(input, attr);
    expect(output.shape).toEqual([3, 4]);
    expect(output.data).toEqual(
      new Float32Array([0, 0, 1, 1.2, 0, 0, 2.3, 3.4, 0, 0, 4.5, 5.7])
    );
  });

  test("Test reflect padding", () => {
    attr.pads = [1, 1, 1, 1];
    attr.mode = "reflect";
    const output = handlePadding(input, attr);

    expect(output.shape).toEqual([5, 4]);
    expect(output.data).toEqual(
      new Float32Array([
        3.4, 2.3, 3.4, 2.3, 1.2, 1, 1.2, 1, 3.4, 2.3, 3.4, 2.3, 5.7, 4.5, 5.7,
        4.5, 3.4, 2.3, 3.4, 2.3,
      ])
    );
  });

  test("Test edge padding", () => {
    attr.pads = [0, 2, 0, 0];
    attr.mode = "edge";
    const output = handlePadding(input, attr);

    expect(output.shape).toEqual([3, 4]);
    expect(output.data).toEqual(
      new Float32Array([1, 1, 1, 1.2, 2.3, 2.3, 2.3, 3.4, 4.5, 4.5, 4.5, 5.7])
    );
  });
});
