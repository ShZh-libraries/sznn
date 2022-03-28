import { Tensor, TensorBuilder } from "../tensor";

// axis=-1 implementation
// TODO: forward_inplace
export function handleSoftmax(inputs: Tensor[]): Tensor[] {
    return [forward(inputs[0])];
}

export function forward(input: Tensor): Tensor {
    const resultShape = input.shape.slice();
    const result = TensorBuilder.withShape(resultShape);

    const dataChannelSize = input.shape[2] * input.shape[3];
    const dataSize = input.shape[1] * dataChannelSize;

    for (let n = 0; n < input.shape[0]; n++) {
        for (let c = 0; c < input.shape[1]; c++) {
            for (let y = 0; y < input.shape[2]; y++) {
                let off = n * dataSize + c * dataChannelSize + y * input.shape[3];

                let max = input.data[off];
                for (let x = 0; x < input.shape[3]; x++) {
                    if (max < input.data[off + x]) {
                        max = input.data[off + x];
                    }
                }

                let sum = 0;
                for (let x = 0; x < input.shape[3]; x++) {
                    // Partial inplace
                    input.data[off + x] = Math.exp(input.data[off + x] - max);
                    sum += input.data[off + x];
                }

                for (let x = 0; x < input.shape[3]; x++) {
                    result.data[off + x] = input.data[off + x] / sum;
                }
            }
        }
    }

    return result;
}