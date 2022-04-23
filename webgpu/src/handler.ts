import { onnx } from "onnx-proto";
import { getConvAttr } from "../../core/attr/conv";
import { getPaddingAttr } from "../../core/attr/padding";
import { getPoolingAttr } from "../../core/attr/pooling";
import { getGPUDevice } from "./gpu";
import { handleBatchNorm } from "./layers/batchnorm";
import { handleConcat } from "./layers/concat";
import { handleConv } from "./layers/conv";
import { handleDropout } from "./layers/dropout";
import { handlePadding } from "./layers/padding";
import { handleAvgPool2D, handleGlobalAvgPool, handleMaxPool2D } from "./layers/pooling";
import { handleLeakyRelu, handleRelu } from "./layers/relu";
import { handleReshape } from "./layers/reshape";
import { handleShape } from "./layers/shape";
import { Tensor } from "./tensor";

export async function handle(
    opType: string,
    inputs: Tensor[],
    attrs: onnx.AttributeProto[]
): Promise<Tensor | Tensor[]> {
    let output: Tensor | Tensor[];
    
    const device = await getGPUDevice()
    switch (opType) {
        case "Conv": {
            const attr = getConvAttr(attrs);
            output = await handleConv(device!, attr, inputs[0], inputs[1], inputs[2]);
            break;
        }
        case "BatchNormalization": {
            output = await handleBatchNorm(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                device!
            );
            break;
        }
        case "Concat": {
            let axis = attrs[0].i as number;
            output = handleConcat(inputs, axis);
            break;
        }
        case "Dropout": {
            output = handleDropout(inputs[0]);
            break;
        }
        case "Pad": {
            const attr = getPaddingAttr(attrs);
            output = await handlePadding(inputs[0], attr, device!);
            break;
        }
        case "MaxPool": {
            const attr = getPoolingAttr(attrs);
            output = await handleMaxPool2D(inputs[0], attr, device!);
            break;
        }
        case "AveragePool": {
            const attr = getPoolingAttr(attrs);
            output = await handleAvgPool2D(inputs[0], attr, device!);
            break;
        }
        case "GlobalAveragePool": {
            output = await handleGlobalAvgPool(inputs[0], device!);
            break;
        }
        case "Relu": {
            output = await handleRelu(inputs[0], device!);
            break;
        }
        case "LeakyRelu": {
            const alpha = attrs[0].f;
            output = await handleLeakyRelu(inputs[0], alpha, device!);
            break;
        }
        case "Shape": {
            output = handleShape(inputs);
            break;
        }
        case "Reshape": {
            const shape = Array.from(inputs[1].data);
            output = handleReshape(inputs[0], shape);
            break;
        }
        default:
            throw new Error(`Unknown op type ${opType}!`);
    }

    // console.log(opType, output);

    return output;
}