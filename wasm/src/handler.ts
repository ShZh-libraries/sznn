import { onnx } from "onnx-proto";
import { getConvAttr } from "../../core/attr/conv";
import { getPoolingAttr } from "../../core/attr/pooling";
import {
  Tensor,
  handleAvgPool2D,
  handleBatchNorm,
  handleConcat,
  handleConv,
  handleDropout,
  handleGlobalAvgPool,
  handleLeakyRelu,
  handleMaxPool2D,
  handleRelu,
  handleReshape,
  handleShape,
  handleAbs,
  handleACos,
  handleACosh,
  handleASin,
  handleASinh,
  handleATan,
  handleATanh,
  handleCeil,
  handleFloor,
  handleRound,
  handleCos,
  handleCosh,
  handleIdentity,
  handleLog,
  handleNeg,
  handleSign,
  handleSin,
  handleSinh,
  handleSqrt,
  handleSigmoid,
  handleTan,
  handleTanh,
  handleAdd,
  handleSub,
  handleMul,
  handleDiv,
} from "./rs/pkg";
import { tensorList } from "./utils";

export function handle(
  opType: string,
  inputs: Tensor[],
  attrs: onnx.AttributeProto[]
): Tensor | Tensor[] {
  let output: Tensor | Tensor[];

  switch (opType) {
    case "Conv": {
      const attr = getConvAttr(attrs);
      output = handleConv(
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.pads[0],
        attr.pads[1],
        attr.pads[2],
        attr.pads[3],
        attr.strides[0],
        attr.strides[1],
        inputs[0],
        inputs[1],
        inputs[2]
      );
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
      output = handleAbs(inputs[0]);
      break;
    }
    case "Acos": {
      output = handleACos(inputs[0]);
      break;
    }
    case "Acosh": {
      output = handleACosh(inputs[0]);
      break;
    }
    case "Asin": {
      output = handleASin(inputs[0]);
      break;
    }
    case "Asinh": {
      output = handleASinh(inputs[0]);
      break;
    }
    case "Atan": {
      output = handleATan(inputs[0]);
      break;
    }
    case "Atanh": {
      output = handleATanh(inputs[0]);
      break;
    }
    case "Ceil": {
      output = handleCeil(inputs[0]);
      break;
    }
    case "Floor": {
      output = handleFloor(inputs[0]);
      break;
    }
    case "Round": {
      output = handleRound(inputs[0]);
      break;
    }
    case "Cos": {
      output = handleCos(inputs[0]);
      break;
    }
    case "Cosh": {
      output = handleCosh(inputs[0]);
      break;
    }
    case "Identity": {
      output = handleIdentity(inputs[0]);
      break;
    }
    case "Log": {
      output = handleLog(inputs[0]);
      break;
    }
    case "Neg": {
      output = handleNeg(inputs[0]);
      break;
    }
    case "Sign": {
      output = handleSign(inputs[0]);
      break;
    }
    case "Sin": {
      output = handleSin(inputs[0]);
      break;
    }
    case "Sinh": {
      output = handleSinh(inputs[0]);
      break;
    }
    case "Sqrt": {
      output = handleSqrt(inputs[0]);
      break;
    }
    case "Sigmoid": {
      output = handleSigmoid(inputs[0]);
      break;
    }
    case "Tan": {
      output = handleTan(inputs[0]);
      break;
    }
    case "Tanh": {
      output = handleTanh(inputs[0]);
      break;
    }
    case "Add": {
      output = handleAdd(inputs[0], inputs[1]);
      break;
    }
    case "Sub": {
      output = handleSub(inputs[0], inputs[1]);
      break;
    }
    case "Mul": {
      output = handleMul(inputs[0], inputs[1]);
      break;
    }
    case "Div": {
      output = handleDiv(inputs[0], inputs[1]);
      break;
    }
    case "Concat": {
      const axis = attrs[0].i as number;
      const list = tensorList(inputs);
      output = handleConcat(list, axis);
      break;
    }
    case "Dropout": {
      output = handleDropout(inputs[0]);
      break;
    }
    case "MaxPool": {
      const attr = getPoolingAttr(attrs);
      output = handleMaxPool2D(
        inputs[0],
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.pads[0],
        attr.pads[1],
        attr.pads[2],
        attr.pads[3],
        attr.strides[0],
        attr.strides[1]
      );
      break;
    }
    case "AveragePool": {
      const attr = getPoolingAttr(attrs);
      output = handleAvgPool2D(
        inputs[0],
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.pads[0],
        attr.pads[1],
        attr.pads[2],
        attr.pads[3],
        attr.strides[0],
        attr.strides[1]
      );
      break;
    }
    case "GlobalAveragePool": {
      output = handleGlobalAvgPool(inputs[0]);
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
    case "Shape": {
      output = handleShape(inputs[0]);
      break;
    }
    case "Reshape": {
      output = handleReshape(inputs[0], inputs[1]);
      break;
    }
    default:
      throw new Error(`Unknown op type ${opType}!`);
  }

  // console.log(opType, output.toArray(), output.shapeToArray());

  return output;
}
