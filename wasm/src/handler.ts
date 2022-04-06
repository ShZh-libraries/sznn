import { onnx } from "onnx-proto";
import { getConvAttr } from "../../core/attr/conv";
import { getPoolingAttr } from "../../core/attr/pooling";
import { handle_avgpool_2d, handle_batchnorm, handle_concat, handle_conv, handle_dropout, handle_global_avgpool, handle_maxpool_2d, handle_relu, Tensor, TensorList, handle_shape, handle_reshape, handle_leaky_relu } from "./rs/pkg";

function Inputs2TensorList(inputs: Tensor[]): TensorList {
    let list = new TensorList();
    for (let input of inputs) {
        list.append(input);
    }

    return list;
}

export function handle(
    opType: string,
    inputs: Tensor[],
    attrs: onnx.AttributeProto[]
) : Tensor | Tensor[] {
    let output: Tensor | Tensor[];

    switch(opType) {
        case "Conv": {
            const attr = getConvAttr(attrs);
            output = handle_conv(
                attr.kernelShape[0], attr.kernelShape[1],
                attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3],
                attr.strides[0], attr.strides[1], 
                inputs[0], inputs[1], inputs[2]
            );
            break;
        }
        case "BatchNormalization": {
            output = handle_batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
            break;
        }
        case "Concat": {
            const axis = attrs[0].i as number;
            const list = Inputs2TensorList(inputs);
            output = handle_concat(list, axis);
            break;
        }
        case "Dropout": {
            output = handle_dropout(inputs[0]);
            break;
        }
        case "MaxPool": {
            const attr = getPoolingAttr(attrs);
            output = handle_maxpool_2d(
                inputs[0],
                attr.kernelShape[0], attr.kernelShape[1],
                attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3],
                attr.strides[0], attr.strides[1]
            );
            break;
        }
        case "AveragePool": {
            const attr = getPoolingAttr(attrs);
            output = handle_avgpool_2d(
                inputs[0],
                attr.kernelShape[0], attr.kernelShape[1],
                attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3],
                attr.strides[0], attr.strides[1]
            );
            break;
        }
        case "GlobalAveragePool": {
            output = handle_global_avgpool(inputs[0]);
            break;
        }
        case "Relu": {
            output = handle_relu(inputs[0]);
            break;
        }
        case "LeakyRelu": {
            const alpha = attrs[0].f;
            output = handle_leaky_relu(inputs[0], alpha);
            break;
        }
        case "Shape": {
            output = handle_shape(inputs[0]);
            break;
        }
        case "Reshape": {
            output = handle_reshape(inputs[0], inputs[1]);
            break;
        }
        default:
            throw new Error(`Unknown op type ${opType}!`);
    }

    console.log(opType, output.toArray(), output.shapeToArray());


    return output;
}