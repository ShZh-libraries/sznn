import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export function handleDropout(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const data = inputs[0];
    const ratio = attributes[0].f;
    const scale = 1 / (1 - ratio);
    
    let result = forward(data, scale);

    // Mask is not needed in inference
    return [result, new Tensor()];
}

export function forward(input: Tensor, scale: number): Tensor {
    let result = TensorBuilder.withShape(input.shape);

    for (let index = 0; index < input.data.length; index++) {
        result.data[index] = input.data[index] * scale;
    }

    return result;
}