import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class PoolingAttr {
    kernelShape: number[] = [];
    pads: number[] = [0, 0, 0, 0];
    strides: number[] = [1, 1];
}

function handleAttribute(attributes: onnx.AttributeProto[]): PoolingAttr {
    let result: PoolingAttr = new PoolingAttr();
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

export function handleMaxPool2D(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const maxPoolingAttr = handleAttribute(attributes);
    const result = forwardMaxPool2D(inputs[0], maxPoolingAttr);
    
    return [result];
}

export function handleAvgPool2D(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const avgPoolingAttr = handleAttribute(attributes);
    const result = forwardAvgPool2D(inputs[0], avgPoolingAttr);

    return [result];
}

export function forwardMaxPool2D(data: Tensor, maxPoolingAttr: PoolingAttr): Tensor {
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

export function forwardAvgPool2D(data: Tensor, avgPoolingAttr: PoolingAttr): Tensor {
    // Calculate shape
    const maxY = data.shape[2] + avgPoolingAttr.pads[0] + avgPoolingAttr.pads[2] - avgPoolingAttr.kernelShape[0];
    const maxX = data.shape[3] + avgPoolingAttr.pads[1] + avgPoolingAttr.pads[3] - avgPoolingAttr.kernelShape[1];
    const resultHeight = Math.floor(maxY / avgPoolingAttr.strides[0]) + 1;
    const resultWidth = Math.floor(maxX / avgPoolingAttr.strides[1]) + 1;

    const kernelSize = avgPoolingAttr.kernelShape[0] * avgPoolingAttr.kernelShape[1];
    const dataChannelSize = data.shape[2] * data.shape[3];
    const dataSize = data.shape[1] * dataChannelSize;
    
    const result = TensorBuilder.withShape([data.shape[0], data.shape[1], resultHeight, resultWidth]);
    let resultIndex = 0;
    for (let n = 0; n < data.shape[0]; n++) {
        for (let c = 0; c < data.shape[1]; c++) {
            for (let y = 0; y <= maxY; y += avgPoolingAttr.strides[0]) {
                for (let x = 0; x <= maxX; x += avgPoolingAttr.strides[1]) {
                    const realY = y - avgPoolingAttr.pads[0];
                    const realX = x - avgPoolingAttr.pads[1];

                    let sum = 0;
                    for (let ky = 0; ky < avgPoolingAttr.kernelShape[0]; ky++) {
                        for (let kx = 0; kx < avgPoolingAttr.kernelShape[1]; kx++) {
                            const dataY = realY + ky;
                            const dataX = realX + kx;

                            if (dataY >= 0 && dataY < data.shape[2] && dataX >= 0 && dataX < data.shape[3]) {
                                sum += data.data[n * dataSize + c * dataChannelSize + dataY * data.shape[3] + dataX];
                            }
                        }
                    }

                    result.data[resultIndex++] = sum / kernelSize;
                }
            }
        }
    }

    return result;
}
