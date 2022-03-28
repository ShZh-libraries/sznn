import { Tensor, TensorBuilder } from "../tensor";

export function handleReshape(inputs: Tensor[]): Tensor[] {
    return [forward(inputs[0], Array.from(inputs[1].data))];
}

// TODO: forward inplace!
export function forward(input: Tensor, shape: number[]): Tensor {
    // Deal with zero
    for (let index = 0; index < shape.length; index++) {
        if (shape[index] == 0) {
            shape[index] = input.shape[index];
        }
    }

    // Deal with negative numbers
    const negativeIndex = shape.findIndex(x => x == -1);
    if (negativeIndex != -1) {
        const remainedLength = shape.filter(x => x != -1).reduceRight((x, y) => x * y);
        const remainedSize = input.data.length / remainedLength;
        shape[negativeIndex] = remainedSize;
    }

    const result = TensorBuilder.withAllArgs(input.data, shape, input.dtype);
    return result;
}