import { forward, SliceAttr } from "../js/layers/slice";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backend for slice layer", () => {
  test("Test simple slice layer", () => {
    const input = TensorBuilder.withData([1, 3, 255, 255]);
    let sliceAttr = new SliceAttr();
    sliceAttr.axes = [0];
    sliceAttr.starts = [2];
    sliceAttr.ends = [4];

    const output = forward(input, sliceAttr);

    expect(output.shape).toEqual([2]);
    expect(output.data).toEqual(new Float32Array([255, 255]));
  });
});
