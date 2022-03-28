import { Tensor, TensorBuilder } from "../tensor";

// The default axis is 1
export function handleSoftmax(inputs: Tensor[]): Tensor[] {
    return [forward(inputs[0])];
}

// For 1 x C x W x H tensor:
export function forward(input: Tensor): Tensor {
    const resultShape = input.shape.slice();
    const result = TensorBuilder.withShape(resultShape);

    let max = input.data[0];
    for (let c = 0; c < input.shape[1]; c++) {
        if (max < input.data[c]) {
            max = input.data[c];
        }
    }

    let sum = 0;
    for (let c = 0; c < input.shape[1]; c++) {
        // Partial inplace
        input.data[c] = Math.exp(input.data[c] - max);
        sum += input.data[c];
    }

    for (let c = 0; c < input.shape[1]; c++) {
        result.data[c] = input.data[c] / sum;
    }

    return result;
}
