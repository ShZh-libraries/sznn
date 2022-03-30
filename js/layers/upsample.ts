import { Tensor, TensorBuilder } from "../tensor";

// This operator is already deprecated
// No need to analysis attribute
export function handleUpSample(inputs: Tensor[]): Tensor[] {
    const output = forward(inputs[0], inputs[1]);
    return [output];
}

export function forward(input: Tensor, scales: Tensor): Tensor {
    // Calculate shape
    let outputShape = [];
    for (let shapeIndex = 0; shapeIndex < input.ndim; shapeIndex++) {
        outputShape.push(input.shape[shapeIndex] * scales.data[shapeIndex]);
    }
    let output = TensorBuilder.withShape(outputShape);

    let index = 0;
    for (let n = 0; n < input.shape[0]; n++) {
        for (let scaleN = 0; scaleN < scales.data[0]; scaleN++) {
            for (let c = 0; c < input.shape[1]; c++) {
                for (let scaleC = 0; scaleC < scales.data[1]; scaleC++) {
                    for (let y = 0; y < input.shape[2]; y++) {
                        for (let scaleH = 0; scaleH < scales.data[2]; scaleH++) {
                            for (let x = 0; x < input.shape[3]; x++) {
                                for (let scaleW = 0; scaleW < scales.data[3]; scaleW++) {
                                    output.data[index++] = input.atLoc([n, c, y, x]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}