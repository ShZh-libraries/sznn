import { PaddingAttr } from "../../../common/attr/padding";
import { Tensor, TensorBuilder } from "../tensor";

function isInPadding(
  outputLoc: number[],
  inputShape: number[],
  shapeOffset: number,
  attr: PaddingAttr
) {
  for (let index = 0; index < attr.pads.length / 2; index++) {
    const curIdx = shapeOffset + index;
    if (
      outputLoc[curIdx] < attr.pads[index] ||
      outputLoc[curIdx] >= inputShape[curIdx] + attr.pads[index]
    ) {
      return true;
    }
  }

  return false;
}

function toInputIdx(
  outputLoc: number[],
  shapeOffset: number,
  attr: PaddingAttr
) {
  let inputLoc = outputLoc.slice();
  for (let index = 0; index < attr.pads.length / 2; index++) {
    const curIdx = shapeOffset + index;
    inputLoc[curIdx] = outputLoc[curIdx] - attr.pads[index];
  }

  return inputLoc;
}

export function handlePadding(input: Tensor, attr: PaddingAttr): Tensor {
  let outputShape = input.shape.slice();
  const shapeOffset = input.shape.length - attr.pads.length / 2;
  for (let index = 0; index < attr.pads.length / 2; index++) {
    outputShape[shapeOffset + index] =
      attr.pads[index] +
      input.shape[shapeOffset + index] +
      attr.pads[index + attr.pads.length / 2];
  }
  let output = TensorBuilder.withShape(outputShape);

  // Passive mode
  for (let index = 0; index < output.data.length; index++) {
    const outputLoc = output.indexToLoc(index);

    if (isInPadding(outputLoc, input.shape, shapeOffset, attr)) {
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
      const inputLoc = toInputIdx(outputLoc, shapeOffset, attr);
      const inputIdx = input.locToIndex(inputLoc);

      output.data[index] = input.data[inputIdx];
    }
  }

  return output;
}
