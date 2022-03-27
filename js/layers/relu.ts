import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export function handleLeakyRelu(input: Tensor[], attribute: onnx.AttributeProto[]): Tensor[] {
    const data = input[0];
    const alpha = attribute[0].f;
    let result = forward(data, alpha);

    return [result];
}

export function forward(data: Tensor, alpha: number): Tensor {
    let result = TensorBuilder.withShape(data.shape);

    for (let index = 0; index < data.data.length; index++) {
        result.data[index] = data.data[index] > 0
            ? data.data[index]
            : alpha * data.data[index];
    }

    return result;
}