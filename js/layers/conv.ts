import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class ConvAttr {
    autoPad: string = "NOTSET";
    dilations: number[] = [];
    group: number = 1;
    kernelShape: number[] = [];
    pads: number[] = [0, 0, 0, 0];
    strides: number[] = [1, 1];
}

function handleAttribute(attributes: onnx.AttributeProto[]): ConvAttr {
    let result: ConvAttr = new ConvAttr();
    for (const attribute of attributes) {
        switch (attribute.name) {
            case "dilations":
                result.dilations = attribute.ints as number[];
                break;
            case "group":
                result.group = attribute.i as number;
                break;
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
                throw new Error(`Unknwo attribute ${attribute.name} in Convolutional layer!!`);
        }
    }

    return result;
}

export function handleConv(input: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const convAttr = handleAttribute(attributes);
    const result = forward(input[0], input[1], convAttr);

    return result;
}

export function forward(data: Tensor, weight: Tensor, convAttr: ConvAttr): Tensor[] {
    // Calculate shape
    const resultSize = data.shape[0];
    const resultChannel = weight.shape[0] / convAttr.group;
    const kernelHeight = weight.shape[2];
    const kernelWidth = weight.shape[3];

    const maxY = data.shape[2] + convAttr.pads[0] + convAttr.pads[2] - kernelHeight;
    const maxX = data.shape[3] + convAttr.pads[1] + convAttr.pads[3] - kernelWidth;
    const resultHeight = Math.floor(maxY / convAttr.strides[1]) + 1;
    const resultWidth = Math.floor(maxX / convAttr.strides[0]) + 1;
    const resultShape = [resultSize, resultChannel, resultHeight, resultWidth];
    const result = TensorBuilder.withShape(resultShape);
 
    // Do convolution
    const kernelChannelSize = convAttr.kernelShape[0] * convAttr.kernelShape[1];
    const kernelSize = kernelChannelSize * weight.shape[1];
    const dataChannelSize = data.shape[2] * data.shape[3];
    const dataSize = dataChannelSize * data.shape[1];

    let resultIndex = 0;
    for (let n = 0; n < resultSize; n++) {
        for (let c = 0; c < resultChannel; c++) {
            for (let y = 0; y <= maxY; y += convAttr.strides[1]) {
                for (let x = 0; x <= maxX; x += convAttr.strides[0]) {
                    const realY = y - convAttr.pads[0];
                    const realX = x - convAttr.pads[1];

                    let sum = 0;
                    for (let ky = 0; ky < convAttr.kernelShape[1]; ky++) {
                        for (let kx = 0; kx < convAttr.kernelShape[0]; kx++) {
                            const dataX = realX + kx;
                            const dataY = realY + ky;

                            if (dataX >= 0 && dataX < data.shape[3] && dataY >= 0 && dataY < data.shape[2]) {
                                for (let kc = 0; kc < weight.shape[1]; kc++) {
                                    const kernelIndex = c * kernelSize + kc * kernelChannelSize + ky * convAttr.kernelShape[0] + kx;
                                    const dataIndex = n * dataSize + kc * dataChannelSize + dataY * data.shape[3] + dataX;
        
                                    const kernelValue = weight.data[kernelIndex];
                                    const dataValue = data.data[dataIndex];
        
                                    sum += kernelValue * dataValue;
                                }
                            }
                        }
                    }
                    result.data[resultIndex++] = sum;
                }
            }
        }
    }

    return [result];
}

// export function forward(data: Tensor, weight: Tensor, convAttr: ConvAttr): Tensor[] {
//     // Calculate shape
//     const resultSize = data.shape[0];
//     const resultChannel = weight.shape[0];
//     const kernelWidth = weight.shape[2];
//     const kernelHeight = weight.shape[3];

//     const resultWidth = Math.floor(data.shape[2] + convAttr.pads[0] + convAttr.pads[2] - kernelWidth) / convAttr.strides[0] + 1;
//     const resultHeight = Math.floor(data.shape[3] + convAttr.pads[1] + convAttr.pads[3] - kernelHeight) / convAttr.strides[1] + 1;
//     const resultShape = [resultSize, resultChannel, resultWidth, resultHeight];
//     const result = TensorBuilder.withShape(resultShape);

//     // Do convolution
//     const kernelSize = kernelWidth * kernelHeight;
//     for (let y = 0; y < resultHeight; y++) {
//         const tempTensor = im2row(data, convAttr, y * convAttr.strides[1]);
//         for (let x = 0; x < resultWidth; x++) {
//             for (let c = 0; c < resultChannel; c++) {
//                 let sum = 0;
//                 for (let i = 0; i < kernelSize; i++) {
//                     sum += weight.data[c * kernelSize + i] as number 
//                 }
//             }
//         }
//     }

//     return [result];
// }

// export function im2row(data: Tensor, convAttr: ConvAttr, y: number): Tensor {
//     const channel = data.shape[1];
//     const width = data.shape[2];

//     for (let i = 0; i < channel; i++) {
//         for (let j = 0; j < convAttr.kernelShape[1]; j++) {
//             for (let x = 0; x < width; x += convAttr.strides[0]) {
//                 for (let k = 0; k < convAttr.kernelShape[0]; x++) {
//                     const srcX = x - convAttr.pads[0] + k;
//                     const srcY = y - convAttr.pads[1] + j;

//                     let target = 0;
//                     if (srcX > 0 && srcY > 0 && srcX < data.shape[2] && srcY < data.shape[3]) {
//                         target = data.data[1] as number;
//                     }
//                 }
//             }
//         }
//     }
// }