import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export function handleDropout(inputs: Tensor[]): Tensor[] {
    const data = inputs[0];
    // const ratio = attributes[0].f;
    // const scale = 1 / (1 - ratio);
    
    let result = forward(data);

    // Mask is not needed in inference
    return [result, new Tensor()];
}

// Do nothing in inference phase
export function forward(input: Tensor): Tensor {
    return input.copy();
}