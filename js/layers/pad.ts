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

// export function handlePadding(inputs: Tensor[], attributes: onnx.AttributeProto[]): Tensor[] {
//     const attr = handleAttributes(attributes);
// }

export function forwardConstantPad(input: Tensor, attr: PaddingAttr): Tensor {
    let resultShape = [];
    for (let index = 0; index < input.ndim; index++) {
        resultShape.push(attr.pads[index] + input.shape[index] + attr.pads[index + input.ndim]);
    }
    let result = TensorBuilder.withShape(resultShape);

    // Passive mode
    for (let index = 0; index < result.data.length; index++) {
        const outputLoc = result.getLoc(index);
        
        if (outputLoc.some((loc, dim) => 
            loc < attr.pads[dim] || 
            loc >= attr.pads[dim] + input.shape[dim]
        )) {
            result.data[index] = 0;
        } else {
            const inputLoc = outputLoc.map((loc, dim) => loc - attr.pads[dim]);
            const inputIdx = input.atLoc(inputLoc);
            
            result.data[index] = input.data[inputIdx];
        }
    }

    return result;
}