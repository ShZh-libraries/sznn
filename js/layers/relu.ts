import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export function handleRelu(inputs: Tensor[]): Tensor[] {
    const data = inputs[0];
    let result = forwardRelu(data);

    return [result];
}

export function handleLeakyRelu(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const data = inputs[0];
    const alpha = attributes[0].f;
    let result = forwardLeakRelu(data, alpha);

    return [result];
}

export function forwardRelu(data: Tensor): Tensor {
    let result = TensorBuilder.withShape(data.shape);

    for (let index = 0; index < data.data.length; index++) {
        result.data[index] = data.data[index] > 0? data.data[index] : 0;
    }

    return result;
}


export function forwardLeakRelu(data: Tensor, alpha: number): Tensor {
    let result = TensorBuilder.withShape(data.shape);

    for (let index = 0; index < data.data.length; index++) {
        result.data[index] = data.data[index] > 0
            ? data.data[index]
            : alpha * data.data[index];
    }

    return result;
}