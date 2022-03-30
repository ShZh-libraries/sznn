import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "../tensor";

export class PaddingAttr {
    mode: string = "constant";
    pads: number[] = [];
}

// For ONNX version 2
export function handleAttributes(attributes: onnx.AttributeProto[]): PaddingAttr {
    let attr = new PaddingAttr();

    for (let attribute of attributes) {
        switch (attribute.name) {
            case "mode":
                attr.mode = new TextDecoder().decode(attribute.s);
                break;
            case "pads":
                attr.pads = attribute.ints as number[];
                break;
            default:
                throw new Error(`Unrecognize attribute name ${attribute.name}!!`);
        }
    }

    return attr;
}

export function handlePadding(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
    const paddingAttr = handleAttributes(attributes);
    const output = forward(inputs[0], paddingAttr);

    return [output];
}

export function forward(input: Tensor, attr: PaddingAttr): Tensor {
    let outputShape = [];
    for (let index = 0; index < input.ndim; index++) {
        outputShape.push(attr.pads[index] + input.shape[index] + attr.pads[index + input.ndim]);
    }
    let output = TensorBuilder.withShape(outputShape);

    // Passive mode
    for (let index = 0; index < output.data.length; index++) {
        const outputLoc = output.indexToLoc(index);

        if (outputLoc.some((loc, dim) => 
            loc < attr.pads[dim] || 
            loc >= attr.pads[dim] + input.shape[dim]
        )) {
            switch (attr.mode) {
                case "constant":
                    output.data[index] = 0;
                    break;
                case "reflect": {
                    let inputLoc = [];
                    for (let i = 0; i < output.ndim; i++) {
                        if (outputLoc[i] < attr.pads[i]) {
                            inputLoc.push((attr.pads[i] - outputLoc[i]) % input.shape[i]);
                        } else if (outputLoc[i] >= attr.pads[i] + input.shape[i]) {
                            let resultIdx = (2 * (attr.pads[i] + input.shape[i] - 1) - outputLoc[i] - attr.pads[i]) % input.shape[i];
                            if (resultIdx < 0) {
                                resultIdx += input.shape[i];
                            }
                            inputLoc.push(resultIdx);
                        } else {
                            inputLoc.push(outputLoc[i] - attr.pads[i]);
                        }
                    }
                    
                    const inputIdx = input.locToIndex(inputLoc);
                    output.data[index] = input.data[inputIdx];
                    break;
                }
                case "edge": {
                    let inputLoc = [];
                    for (let i = 0; i < output.ndim; i++) {
                        if (outputLoc[i] < attr.pads[i]) {
                            inputLoc.push(0);
                        } else if (outputLoc[i] >= attr.pads[i] + input.shape[i]) {
                            inputLoc.push(input.shape[i] - 1);
                        } else {
                            inputLoc.push(outputLoc[i] - attr.pads[i]);
                        }
                    }
                    const inputIdx = input.locToIndex(inputLoc);
                    output.data[index] = input.data[inputIdx];
                    break;
                }
                default:
                    throw new Error(`Padding mode ${attr.mode} not recognized!`);
            }
        } else {
            const inputLoc = outputLoc.map((loc, dim) => loc - attr.pads[dim]);
            const inputIdx = input.locToIndex(inputLoc);
            
            output.data[index] = input.data[inputIdx];
        }
    }

    return output;
}