import { Tensor, TensorBuilder } from "../tensor";

// TODO: attribute like epsilon
export function handleBatchNorm(input: Tensor[]): Tensor[] {
    // Extract weights
    const data = input[0];
    const scale = input[1];
    const bias = input[2];
    const mean = input[3];
    const variance = input[4];

    // Calculate shape
    let result = TensorBuilder.withShape(data.shape);

    const dataLength = data.shape.reduceRight((x, y) => x * y);
    for (let index = 0; index < dataLength; index++) {
            result.data[index] = 
                (data.data[index] - mean.data[index]) / Math.sqrt(variance.data[index]) * scale.data[index] + bias.data[index];
        
    }

    return [result];
}