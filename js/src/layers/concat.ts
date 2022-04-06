import { Tensor, TensorBuilder } from "../tensor";

export function handleConcat(inputs: Tensor[], axis: number): Tensor {
  // Special cases
  if (inputs.length == 1) {
    return inputs[0];
  }
  if (inputs[0].ndim == 1) {
    return concat1D(inputs);
  }
  if (inputs[0].ndim == 2) {
    if (axis == 0) {
      return concat2DAxis0(inputs);
    } else if (axis == 1) {
      return concat2DAxis1(inputs);
    }
  }

  // Transform other tensors to 2D form and concat them with axis=1
  const outputShape = getConcatShape(inputs, axis);
  const transformedTensors = inputs.map((input) => {
    let height, width;
    if (axis != 0) {
      height = input.shape.slice(0, axis).reduceRight((x, y) => x * y);
      width = input.shape.slice(axis).reduceRight((x, y) => x * y);
    } else {
      height = 1;
      width = input.shape.reduceRight((x, y) => x * y);
    }

    // TODO: introduce real tensor pool to avoid copy
    return input.copy().reshape([height, width]);
  });

  const transformedResult = concat2DAxis1(transformedTensors);
  const output = transformedResult.reshape(outputShape);

  return output;
}

function getConcatShape(data: Tensor[], axis: number): number[] {
  let axisSize = 0;
  for (const input of data) {
    axisSize += input.shape[axis];
  }
  let resultShape = data[0].shape.slice();
  resultShape[axis] = axisSize;

  return resultShape;
}

function concat1D(data: Tensor[]): Tensor {
  const resultShape = data.map((x) => x.shape[0]).reduceRight((x, y) => x + y);
  const result = TensorBuilder.withShape([resultShape]);

  let resultIndex = 0;
  for (let input of data) {
    for (let index = 0; index < input.shape[0]; index++) {
      result.data[resultIndex++] = input.data[index];
    }
  }

  return result;
}

function concat2DAxis0(data: Tensor[]): Tensor {
  const resultShape = getConcatShape(data, 0);
  const result = TensorBuilder.withShape(resultShape);

  let resultIndex = 0;
  const height = data[0].shape[0];
  const width = data[0].shape[1];
  for (let input of data) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        result.data[resultIndex++] = input.data[y * width + x];
      }
    }
  }

  return result;
}

function concat2DAxis1(data: Tensor[]): Tensor {
  const resultShape = getConcatShape(data, 1);
  const result = TensorBuilder.withShape(resultShape);

  let resultIndex = 0;
  const height = data[0].shape[0];
  const width = data[0].shape[1];
  for (let y = 0; y < height; y++) {
    for (let input of data) {
      for (let x = 0; x < width; x++) {
        result.data[resultIndex++] = input.data[y * width + x];
      }
    }
  }

  return result;
}
