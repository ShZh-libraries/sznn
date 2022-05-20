import { onnx } from "onnx-proto";
import { getConvAttr } from "../../common/attr/conv";
import { getSliceAttr } from "../../common/attr/slice";
import { getPoolingAttr } from "../../common/attr/pooling";
import { getPaddingAttr } from "../../common/attr/padding";
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
  handleAcos,
  handleAcosh,
  handleAsin,
  handleAsinh,
  handleAtan,
  handleAtanh,
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
  handleUpSample,
  handleUnsqueeze,
  handleSlice,
  handleInstanceNorm,
  handleGather,
  handleCast,
  handlePadding,
} from "./rs/pkg";
import { tensorList } from "./utils";
import { TensorBuilder } from "./tensor";

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
      output = handleAcos(inputs[0]);
      break;
    }
    case "Acosh": {
      output = handleAcosh(inputs[0]);
      break;
    }
    case "Asin": {
      output = handleAsin(inputs[0]);
      break;
    }
    case "Asinh": {
      output = handleAsinh(inputs[0]);
      break;
    }
    case "Atan": {
      output = handleAtan(inputs[0]);
      break;
    }
    case "Atanh": {
      output = handleAtanh(inputs[0]);
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
    case "Pad": {
      const attr = getPaddingAttr(attrs);
      output = handlePadding(
        inputs[0],
        attr.pads[2],
        attr.pads[3],
        attr.pads[6],
        attr.pads[7]
      );
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
    case "Slice": {
      const attr = getSliceAttr(attrs);
      output = handleSlice(inputs[0], attr.axes, attr.starts, attr.ends);
      break;
    }
    case "Unsqueeze": {
      const dims = attrs[0].ints as number[];
      output = handleUnsqueeze(inputs[0], dims);
      break;
    }
    case "Upsample": {
      output = handleUpSample(inputs[0], inputs[1]);
      break;
    }
    default:
      throw new Error(`Unknown op type ${opType}!`);
  }

  console.log(opType, output.toArray(), output.shapeToArray());

  return output;
}

function handleConstant(attributes: onnx.AttributeProto[]): Tensor {
  const dims = attributes[0].t!.dims! as number[];
  const outShape = dims.length == 0 ? [1] : dims;
  let buffer = attributes[0].t!.rawData!.buffer.slice(
    attributes[0].t!.rawData!.byteOffset,
    attributes[0].t!.rawData!.byteOffset + attributes[0].t!.rawData!.byteLength
  );

  let outData;
  switch (attributes[0].t!.dataType) {
    case 1:
      outData = new Float32Array(buffer);
      break;
    case 2:
      outData = new Uint8Array(buffer);
      break;
    case 3:
      outData = new Int8Array(buffer);
      break;
    case 4:
      outData = new Uint16Array(buffer);
      break;
    case 5:
      outData = new Int16Array(buffer);
      break;
    case 6:
      outData = new Int32Array(buffer);
      break;
    case 7: {
      const temp = new BigInt64Array(buffer);
      outData = new Int32Array([Number(temp[0])]);
      break;
    }
    case 11:
      outData = new Float64Array(buffer);
      break;
    case 12:
      outData = new Uint32Array(buffer);
      break;
    case 13: {
      const temp = new BigUint64Array(buffer);
      outData = new Uint32Array([Number(temp[0])]);
      break;
    }
    default:
      throw Error("Data type not support in ONNX!!");
  }

  return TensorBuilder.withAllArgs(outData, outShape);
}
