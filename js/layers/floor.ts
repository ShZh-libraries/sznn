import { Tensor, TensorBuilder } from "../tensor";

export function handleFloor(inputs: Tensor[]): Tensor[] {
    return [forward(inputs[0])];
}

// TODO: in place forward
export function forward(input: Tensor): Tensor {
    let result = TensorBuilder.withShape(input.shape);
    
    for (let index = 0; index < result.data.length; index++) {
        result.data[index] = Math.floor(input.data[index]);
    }

    return result;
}