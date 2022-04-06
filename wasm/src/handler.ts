import { onnx } from "onnx-proto";
import { getConvAttr } from "../../core/attr/conv";
import { getPoolingAttr } from "../../core/attr/pooling";
import {
  Tensor,
  TensorList,
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
} from "./rs/pkg";

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
    case "Concat": {
      const axis = attrs[0].i as number;
      const list = Inputs2TensorList(inputs);
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
