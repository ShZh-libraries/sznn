import { onnx } from "onnx-proto";
import { getConvAttr } from "../../core/attr/conv";
import { getPaddingAttr } from "../../core/attr/padding";
import { getPoolingAttr } from "../../core/attr/pooling";
import { getGPUDevice } from "./gpu";
import { handleBatchNorm } from "./layers/batchnorm";
import { handleBinaryOp } from "./layers/binaryop";
import { handleConcat } from "./layers/concat";
import { handleConv } from "./layers/conv";
import { handleDropout } from "./layers/dropout";
import { handlePadding } from "./layers/padding";
import { handleAvgPool2D, handleGlobalAvgPool, handleMaxPool2D } from "./layers/pooling";
import { handleLeakyRelu, handleRelu } from "./layers/relu";
import { handleReshape } from "./layers/reshape";
import { handleShape } from "./layers/shape";
import { handleUnaryOp } from "./layers/unaryop";
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
        case "Abs": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = abs(input[global_id.x])", device!);
            break;
        }
        case "Acos": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = acos(input[global_id.x])", device!);
            break;
        }
        case "Acosh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = acosh(input[global_id.x])", device!);
            break;
        }
        case "Asin": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = asin(input[global_id.x])", device!);
            break;
        }
        case "Asinh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = asinh(input[global_id.x])", device!);
            break;
        }
        case "Atan": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = atan(input[global_id.x])", device!);
            break;
        }
        case "Atanh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = atanh(input[global_id.x])", device!);
            break;
        }
        case "Ceil": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = ceil(input[global_id.x])", device!);
            break;
        }
        case "Floor": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = floor(input[global_id.x])", device!);
            break;
        }
        case "Round": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = round(input[global_id.x])", device!);
            break;
        }
        case "Cos": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = cos(input[global_id.x])", device!);
            break;
        }
        case "Cosh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = cosh(input[global_id.x])", device!);
            break;
        }
        case "Identity": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = input[global_id.x]", device!);
            break;
        }
        case "Log": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = log(input[global_id.x])", device!);
            break;
        }
        case "Neg": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = -input[global_id.x]", device!);
            break;
        }
        case "Sign": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = sign(input[global_id.x])", device!);
            break;
        }
        case "Sin": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = sin(input[global_id.x])", device!);
            break;
        }
        case "Sinh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = sinh(input[global_id.x])", device!);
            break;
        }
        case "Sqrt": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = sqrt(input[global_id.x])", device!);
            break;
        }
        case "Sigmoid": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = 1. / (1. + exp(input[global_id.x]))", device!);
            break;
        }
        case "Tan": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = tan(input[global_id.x])", device!);
            break;
        }
        case "Tanh": {
            output = await handleUnaryOp(inputs[0], "output[global_id.x] = tanh(input[global_id.x])", device!);
            break;
        }
        case "Add": {
            output = await handleBinaryOp(inputs[0], inputs[1], "+", device!);
            break;
        }
        case "Sub": {
            output = await handleBinaryOp(inputs[0], inputs[1], "-", device!);
            break;
        }
        case "Mul": {
            output = await handleBinaryOp(inputs[0], inputs[1], "*", device!);
            break;
        }
        case "Div": {
            output = await handleBinaryOp(inputs[0], inputs[1], "/", device!);
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