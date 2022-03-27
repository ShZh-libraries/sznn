import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class MaxPoolAttr {
    kernelShape: number[] = [];
    pads: number[] = [0, 0, 0, 0];
    strides: number[] = [1, 1];
}

function handleAttribute(attributes: onnx.AttributeProto[]): MaxPoolAttr {
    let result: MaxPoolAttr = new MaxPoolAttr();
    for (const attribute of attributes) {
        switch(attribute.name) {
            case "kernel_shape":
                result.kernelShape = attribute.ints as number[];
                break;
            case "pads":
                result.pads = attribute.ints as number[];
                break;
            case "strides":
                result.strides = attribute.ints as number[];
                break;
            default:
                throw new Error(`Unknown attribute ${attribute.name} in MaxPooling layer!!`);
        }
    }

    return result;
}

export function handleMaxPool(input: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const maxPoolingAttr = handleAttribute(attributes);
    const result = forward(input[0], maxPoolingAttr);
    
    return [result];
}

// 2D version
export function forward(data: Tensor, maxPoolingAttr: MaxPoolAttr): Tensor {
    // Calculate shape
    const maxY = data.shape[2] + maxPoolingAttr.pads[0] + maxPoolingAttr.pads[2] - maxPoolingAttr.kernelShape[0];
    const maxX = data.shape[3] + maxPoolingAttr.pads[1] + maxPoolingAttr.pads[3] - maxPoolingAttr.kernelShape[1];
    const resultHeight = Math.floor(maxY / maxPoolingAttr.strides[0]) + 1;
    const resultWidth = Math.floor(maxX / maxPoolingAttr.strides[1]) + 1;

    const dataChannelSize = data.shape[2] * data.shape[3];
    const dataSize = data.shape[1] * dataChannelSize;
    
    const result = TensorBuilder.withShape([data.shape[0], data.shape[1], resultHeight, resultWidth]);
    let resultIndex = 0;
    for (let n = 0; n < data.shape[0]; n++) {
        for (let c = 0; c < data.shape[1]; c++) {
            for (let y = 0; y <= maxY; y += maxPoolingAttr.strides[0]) {
                for (let x = 0; x <= maxX; x += maxPoolingAttr.strides[1]) {
                    const realY = y - maxPoolingAttr.pads[0];
                    const realX = x - maxPoolingAttr.pads[1];

                    let maxValue = data.data[n * dataSize + c * dataChannelSize + realY * data.shape[3] + realX];
                    for (let ky = 0; ky < maxPoolingAttr.kernelShape[0]; ky++) {
                        for (let kx = 0; kx < maxPoolingAttr.kernelShape[1]; kx++) {
                            const dataY = realY + ky;
                            const dataX = realX + kx;

                            let currentValue = 0;
                            if (dataY >= 0 && dataY < data.shape[2] && dataX >= 0 && dataX < data.shape[3]) {
                                currentValue = data.data[n * dataSize + c * dataChannelSize + dataY * data.shape[3] + dataX];
                            }

                            if (currentValue > maxValue) {
                                maxValue = currentValue;
                            }
                        }
                    }

                    result.data[resultIndex++] = maxValue;
                }
            }
        }
    }

    return result;
}
