import { SliceAttr } from "../../common/attr/slice";
import { handleSlice } from "../src/layers/slice";
import { TensorBuilder } from "../src/tensor";

describe("Test JS backend for slice layer", () => {
  test("Test simple slice layer", () => {
    const input = TensorBuilder.withData([1, 3, 255, 255]);
    let sliceAttr = new SliceAttr();
    sliceAttr.axes = [0];
    sliceAttr.starts = [2];
    sliceAttr.ends = [4];

    const output = handleSlice(input, sliceAttr);

    expect(output.shape).toEqual([2]);
    expect(output.data).toEqual(new Float32Array([255, 255]));
  });
});
