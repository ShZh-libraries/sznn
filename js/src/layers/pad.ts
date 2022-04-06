import { PaddingAttr } from "../../../core/attr/padding";
import { Tensor, TensorBuilder } from "../tensor";


export function handlePadding(input: Tensor, attr: PaddingAttr): Tensor {
  let outputShape = [];
  for (let index = 0; index < input.ndim; index++) {
    outputShape.push(
      attr.pads[index] + input.shape[index] + attr.pads[index + input.ndim]
    );
  }
  let output = TensorBuilder.withShape(outputShape);

  // Passive mode
  for (let index = 0; index < output.data.length; index++) {
    const outputLoc = output.indexToLoc(index);

    if (
      outputLoc.some(
        (loc, dim) =>
          loc < attr.pads[dim] || loc >= attr.pads[dim] + input.shape[dim]
      )
    ) {
      switch (attr.mode) {
        case "constant":
          output.data[index] = 0;
          break;
        case "reflect": {
          let inputLoc = [];
          for (let i = 0; i < output.ndim; i++) {
            if (outputLoc[i] < attr.pads[i]) {
              inputLoc.push((attr.pads[i] - outputLoc[i]) % input.shape[i]);
            } else if (outputLoc[i] >= attr.pads[i] + input.shape[i]) {
              let resultIdx =
                (2 * (attr.pads[i] + input.shape[i] - 1) -
                  outputLoc[i] -
                  attr.pads[i]) %
                input.shape[i];
              if (resultIdx < 0) {
                resultIdx += input.shape[i];
              }
              inputLoc.push(resultIdx);
            } else {
              inputLoc.push(outputLoc[i] - attr.pads[i]);
            }
          }

          const inputIdx = input.locToIndex(inputLoc);
          output.data[index] = input.data[inputIdx];
          break;
        }
        case "edge": {
          let inputLoc = [];
          for (let i = 0; i < output.ndim; i++) {
            if (outputLoc[i] < attr.pads[i]) {
              inputLoc.push(0);
            } else if (outputLoc[i] >= attr.pads[i] + input.shape[i]) {
              inputLoc.push(input.shape[i] - 1);
            } else {
              inputLoc.push(outputLoc[i] - attr.pads[i]);
            }
          }
          const inputIdx = input.locToIndex(inputLoc);
          output.data[index] = input.data[inputIdx];
          break;
        }
        default:
          throw new Error(`Padding mode ${attr.mode} not recognized!`);
      }
    } else {
      const inputLoc = outputLoc.map((loc, dim) => loc - attr.pads[dim]);
      const inputIdx = input.locToIndex(inputLoc);

      output.data[index] = input.data[inputIdx];
    }
  }

  return output;
}
