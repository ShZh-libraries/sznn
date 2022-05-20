import { Tensor, TensorBuilder } from "../tensor";
import { SliceAttr } from "../../../common/attr/slice";

export function handleSlice(input: Tensor, attr: SliceAttr): Tensor {
  if (attr.axes.length == 1 && attr.axes[0] == 0) {
    const outputShape = attr.ends[0] - attr.starts[0];
    const output = TensorBuilder.withShape([outputShape]);
    for (let i = attr.starts[0]; i < attr.ends[0]; i++) {
      output.data[i - attr.starts[0]] = input.data[i];
    }

    return output;
  }

  return new Tensor();
}
