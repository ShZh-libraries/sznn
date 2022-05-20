import { Tensor, TensorBuilder } from "../tensor";

type BinaryFunc = (x: number, y: number) => number;

export function handleBinaryOp(a: Tensor, b: Tensor, op: BinaryFunc): Tensor {
  const outputShape = getBroadcastShape(a.shape, b.shape);
  const output = TensorBuilder.withShape(outputShape);

  const aBroadcastDim = getBroadcastDims(a.shape, outputShape);
  const bBroadcastDim = getBroadcastDims(b.shape, outputShape);

  if (aBroadcastDim.length + bBroadcastDim.length == 0) {
    for (let i = 0; i < output.data.length; i++) {
      output.data[i] = op(a.data[i], b.data[i]);
    }
  } else {
    for (let i = 0; i < output.data.length; i++) {
      const outputLoc = output.indexToLoc(i);

      const aLoc = outputLoc.slice(-a.ndim);
      aBroadcastDim.forEach((dim) => (aLoc[dim] = 0));
      const aIndex = a.locToIndex(aLoc);

      const bLoc = outputLoc.slice(-b.ndim);
      bBroadcastDim.forEach((dim) => (bLoc[dim] = 0));
      const bIndex = b.locToIndex(bLoc);

      output.data[i] = op(a.data[aIndex], b.data[bIndex]);
    }
  }

  return output;
}

function getBroadcastShape(shape1: number[], shape2: number[]): number[] {
  let result = [];
  let resultLength =
    shape1.length > shape2.length ? shape1.length : shape2.length;

  for (let index = 0; index < resultLength; index++) {
    let a = shape1[shape1.length - 1 - index]
      ? shape1[shape1.length - 1 - index]
      : 1;
    let b = shape2[shape2.length - 1 - index]
      ? shape2[shape2.length - 1 - index]
      : 1;

    if (a == 1) {
      result.unshift(b);
    } else if (b == 1) {
      result.unshift(a);
    } else if (a == b) {
      result.unshift(a);
    } else {
      throw new Error("Cannot broadcast!!");
    }
  }

  return result;
}

function getBroadcastDims(shape: number[], resultShape: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    const dim = shape.length - 1 - i;
    const a = shape[dim] || 1;
    const b = resultShape[dim] || 1;
    if (b > 1 && a === 1) {
      result.unshift(dim);
    }
  }
  return result;
}
