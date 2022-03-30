import { Tensor, TensorBuilder } from "../tensor";

export function handleAbs(inputs: Tensor[]): Tensor[] {
    return [forward(inputs[0])];
}

// TODO: forward inplace!!
export function forward(input: Tensor): Tensor {
    let output = TensorBuilder.withShape(input.shape);

    for (let index = 0; index < input.data.length; index++) {
        output.data[index] = Math.abs(input.data[index]);
    }

    return output;
}