import { onnx } from "onnx-proto";
import { Tensor } from "./tensor";
import { handleBatchNorm } from "./layers/batchnorm";
import { handleBinaryOp } from "./layers/binaryop";
import { handleCast } from "./layers/cast";
import { handleConcat } from "./layers/concat";
import { handleConstant } from "./layers/constant";
import { handleConv } from "./layers/conv";
import { handleDropout } from "./layers/dropout";
import { handleGather } from "./layers/gather";
import { handleInstanceNorm } from "./layers/instancenorm";
import { handlePadding } from "./layers/pad";
import { handleAvgPool2D, handleGlobalAvgPool, handleMaxPool2D } from "./layers/pooling";
import { handleLeakyRelu, handleRelu } from "./layers/relu";
import { handleReshape } from "./layers/reshape";
import { handleShape } from "./layers/shape";
import { handleSlice } from "./layers/slice";
import { handleSoftmax } from "./layers/softmax";
import { handleUnaryOp } from "./layers/unaryop";
import { handleUnsqueeze } from "./layers/unsqueeze";
import { handleUpSample } from "./layers/upsample";

export function handle(opType: string, inputs: Tensor[], attrs: onnx.AttributeProto[]): Tensor[] {
    let outputs: Tensor[];

    switch (opType) {
        case "Conv":
            outputs = handleConv(inputs, attrs); 
            break;
        case "BatchNormalization":
            outputs = handleBatchNorm(inputs);
            break;
        case "Abs":
            outputs = handleUnaryOp(inputs, x => Math.abs(x));
            break;
        case "Acos":
            outputs = handleUnaryOp(inputs, x => Math.acos(x));
            break;
        case "Acosh":
            outputs = handleUnaryOp(inputs, x => Math.acosh(x));
            break;
        case "Asin":
            outputs = handleUnaryOp(inputs, x => Math.asin(x));
            break;
        case "Asinh":
            outputs = handleUnaryOp(inputs, x => Math.asinh(x));
            break;
        case "Atan":
            outputs = handleUnaryOp(inputs, x => Math.atan(x));
            break;
        case "Atanh":
            outputs = handleUnaryOp(inputs, x => Math.atanh(x));
            break;
        case "Ceil":
            outputs = handleUnaryOp(inputs, x => Math.ceil(x));
            break;
        case "Floor":
            outputs = handleUnaryOp(inputs, x => Math.floor(x));
            break;
        case "Round":
            outputs = handleUnaryOp(inputs, x => Math.round(x));
            break;
        case "Cos":
            outputs = handleUnaryOp(inputs, x => Math.cos(x));
            break;
        case "Cosh":
            outputs = handleUnaryOp(inputs, x => Math.cosh(x));
            break;
        // case "IsInf":
        //     outputs = handleUnaryOp(inputTensors, x => isFinite(x));
        //     break;
        // case "IsNaN"
        case "Identity":
            outputs = handleUnaryOp(inputs, x => x);
            break;
        case "Log":
            outputs = handleUnaryOp(inputs, x => Math.log(x));
            break;
        case "Neg":
            outputs = handleUnaryOp(inputs, x => -x);
            break;
        // case "Not":
        //     outputs = handleUnaryOp(inputTensors, x => !x);
        //     break;
        case "Sign":
            outputs = handleUnaryOp(inputs, x => Math.sign(x));
            break;
        case "Sin":
            outputs = handleUnaryOp(inputs, x => Math.sin(x));
            break;
        case "Sinh":
            outputs = handleUnaryOp(inputs, x => Math.sinh(x));
            break;
        case "Sqrt":
            outputs = handleUnaryOp(inputs, x => Math.sqrt(x));
            break;
        case "Sigmoid":
            outputs = handleUnaryOp(inputs, x => 1 / (1 + Math.exp(-x)));
            break;
        case "Tan":
            outputs = handleUnaryOp(inputs, x => Math.tan(x));
            break;
        case "Tanh":
            outputs = handleUnaryOp(inputs, x => Math.tanh(x));
            break;
        case "Add":
            outputs = handleBinaryOp(inputs, (x, y) => x + y);
            break;
        case "Sub":
            outputs = handleBinaryOp(inputs, (x, y) => x - y);
            break;
        case "Mul":
            outputs = handleBinaryOp(inputs, (x, y) => x * y);
            break;
        case "Div":
            outputs = handleBinaryOp(inputs, (x, y) => x / y);
            break;
        // case "Equal":
        //     outputs = handleBinaryOp(inputTensors, (x, y) => x == y);
        //     break;
        case "Cast":
            outputs = handleCast(inputs, attrs);
            break;
        case "Constant":
            outputs = handleConstant(attrs);
            break;
        case "Gather":
            outputs = handleGather(inputs);
            break;
        case "InstanceNormalization":
            outputs = handleInstanceNorm(inputs, attrs);
            break;
        case "Pad":
            outputs = handlePadding(inputs, attrs);
            break;
        case "Shape":
            outputs = handleShape(inputs);
            break;
        case "Slice":
            outputs = handleSlice(inputs, attrs);
            break;
        case "Unsqueeze":
            outputs = handleUnsqueeze(inputs, attrs);
            break;
        case "Dropout":
            outputs = handleDropout(inputs);
            break;
        case "Relu":
            outputs = handleRelu(inputs);
            break;
        case "LeakyRelu":
            outputs = handleLeakyRelu(inputs, attrs);
            break;
        case "MaxPool":
            outputs = handleMaxPool2D(inputs, attrs);
            break;
        case "AveragePool":
            outputs = handleAvgPool2D(inputs, attrs);
            break;
        case "GlobalAveragePool":
            outputs = handleGlobalAvgPool(inputs);
            break;
        case "Upsample":
            outputs = handleUpSample(inputs);
            break;
        case "Concat":
            outputs = handleConcat(inputs, attrs);
            break;
        case "Softmax":
            outputs = handleSoftmax(inputs);
            break;
        case "Reshape":
            outputs = handleReshape(inputs);
            break;
        default:
            throw new Error(`Unknown op type ${opType}!`);
    }

    // console.log(opType, outputs);

    return outputs;
}