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
import { handlePadding } from "./layers/padding";
import {
  handleAvgPool2D,
  handleGlobalAvgPool,
  handleMaxPool2D,
} from "./layers/pooling";
import { handleLeakyRelu, handleRelu } from "./layers/relu";
import { handleReshape } from "./layers/reshape";
import { handleShape } from "./layers/shape";
import { handleSlice } from "./layers/slice";
import { handleSoftmax } from "./layers/softmax";
import { handleUnaryOp } from "./layers/unaryop";
import { handleUnsqueeze } from "./layers/unsqueeze";
import { handleUpSample } from "./layers/upsample";
import { getConvAttr } from "../../common/attr/conv";
import { getPaddingAttr } from "../../common/attr/padding";
import { getSliceAttr } from "../../common/attr/slice";
import { getPoolingAttr } from "../../common/attr/pooling";

export function handle(
  opType: string,
  inputs: Tensor[],
  attrs: onnx.AttributeProto[]
): Tensor | Tensor[] {
  let output: Tensor | Tensor[];

  switch (opType) {
    case "Conv": {
      const attr = getConvAttr(attrs);
      output = handleConv(attr, inputs[0], inputs[1], inputs[2]);
      break;
    }
    case "BatchNormalization": {
      output = handleBatchNorm(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4]
      );
      break;
    }
    case "Abs": {
      output = handleUnaryOp(inputs[0], (x) => Math.abs(x));
      break;
    }
    case "Acos": {
      output = handleUnaryOp(inputs[0], (x) => Math.acos(x));
      break;
    }
    case "Acosh": {
      output = handleUnaryOp(inputs[0], (x) => Math.acosh(x));
      break;
    }
    case "Asin": {
      output = handleUnaryOp(inputs[0], (x) => Math.asin(x));
      break;
    }
    case "Asinh": {
      output = handleUnaryOp(inputs[0], (x) => Math.asinh(x));
      break;
    }
    case "Atan": {
      output = handleUnaryOp(inputs[0], (x) => Math.atan(x));
      break;
    }
    case "Atanh": {
      output = handleUnaryOp(inputs[0], (x) => Math.atanh(x));
      break;
    }
    case "Ceil": {
      output = handleUnaryOp(inputs[0], (x) => Math.ceil(x));
      break;
    }
    case "Floor": {
      output = handleUnaryOp(inputs[0], (x) => Math.floor(x));
      break;
    }
    case "Round": {
      output = handleUnaryOp(inputs[0], (x) => Math.round(x));
      break;
    }
    case "Cos": {
      output = handleUnaryOp(inputs[0], (x) => Math.cos(x));
      break;
    }
    case "Cosh": {
      output = handleUnaryOp(inputs[0], (x) => Math.cosh(x));
      break;
    }
    // case "IsInf":
    //     output = handleUnaryOp(inputTensors, x => isFinite(x));
    //     break;
    // case "IsNaN"
    case "Identity": {
      output = handleUnaryOp(inputs[0], (x) => x);
      break;
    }
    case "Log": {
      output = handleUnaryOp(inputs[0], (x) => Math.log(x));
      break;
    }
    case "Neg": {
      output = handleUnaryOp(inputs[0], (x) => -x);
      break;
    }
    // case "Not":
    //     output = handleUnaryOp(inputTensors, x => !x);
    //     break;
    case "Sign": {
      output = handleUnaryOp(inputs[0], (x) => Math.sign(x));
      break;
    }
    case "Sin": {
      output = handleUnaryOp(inputs[0], (x) => Math.sin(x));
      break;
    }
    case "Sinh": {
      output = handleUnaryOp(inputs[0], (x) => Math.sinh(x));
      break;
    }
    case "Sqrt": {
      output = handleUnaryOp(inputs[0], (x) => Math.sqrt(x));
      break;
    }
    case "Sigmoid": {
      output = handleUnaryOp(inputs[0], (x) => 1 / (1 + Math.exp(-x)));
      break;
    }
    case "Tan": {
      output = handleUnaryOp(inputs[0], (x) => Math.tan(x));
      break;
    }
    case "Tanh": {
      output = handleUnaryOp(inputs[0], (x) => Math.tanh(x));
      break;
    }
    case "Add": {
      output = handleBinaryOp(inputs[0], inputs[1], (x, y) => x + y);
      break;
    }
    case "Sub": {
      output = handleBinaryOp(inputs[0], inputs[1], (x, y) => x - y);
      break;
    }
    case "Mul": {
      output = handleBinaryOp(inputs[0], inputs[1], (x, y) => x * y);
      break;
    }
    case "Div": {
      output = handleBinaryOp(inputs[0], inputs[1], (x, y) => x / y);
      break;
    }
    // case "Equal":
    //     output = handleBinaryOp(inputTensors, (x, y) => x == y);
    //     break;
    case "Cast": {
      const to = attrs[0].i as number;
      output = handleCast(inputs[0], to);
      break;
    }
    case "Constant": {
      output = handleConstant(attrs);
      break;
    }
    case "Gather": {
      output = handleGather(inputs[0], inputs[1]);
      break;
    }
    case "InstanceNormalization": {
      const epsilon = attrs[0].f;
      output = handleInstanceNorm(inputs[0], inputs[1], inputs[2], epsilon);
      break;
    }
    case "Pad": {
      const attr = getPaddingAttr(attrs);
      output = handlePadding(inputs[0], attr);
      break;
    }
    case "Shape": {
      output = handleShape(inputs);
      break;
    }
    case "Slice": {
      const attr = getSliceAttr(attrs);
      output = handleSlice(inputs[0], attr);
      break;
    }
    case "Unsqueeze": {
      const dims = attrs[0].ints as number[];
      output = handleUnsqueeze(inputs[0], dims);
      break;
    }
    case "Dropout": {
      output = handleDropout(inputs[0]);
      break;
    }
    case "Relu": {
      output = handleRelu(inputs[0]);
      break;
    }
    case "LeakyRelu": {
      const alpha = attrs[0].f;
      output = handleLeakyRelu(inputs[0], alpha);
      break;
    }
    case "MaxPool": {
      const attr = getPoolingAttr(attrs);
      output = handleMaxPool2D(inputs[0], attr);
      break;
    }
    case "AveragePool": {
      const attr = getPoolingAttr(attrs);
      output = handleAvgPool2D(inputs[0], attr);
      break;
    }
    case "GlobalAveragePool": {
      output = handleGlobalAvgPool(inputs[0]);
      break;
    }
    case "Upsample": {
      output = handleUpSample(inputs[0], inputs[1]);
      break;
    }
    case "Concat": {
      let axis = attrs[0].i as number;
      output = handleConcat(inputs, axis);
      break;
    }
    case "Softmax": {
      output = handleSoftmax(inputs[0]);
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
