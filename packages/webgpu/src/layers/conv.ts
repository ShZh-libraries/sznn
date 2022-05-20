import conv from "./shaders/conv.wgsl";
import { ConvAttr } from "../../../common/attr/conv";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";
import { handlePadding } from "./padding";

export async function handleConv(
  device: GPUDevice,
  attr: ConvAttr,
  input: Tensor,
  weight: Tensor,
  bias?: Tensor
): Promise<Tensor> {
  // Padding first
  const paddingAttr = attr.getPaddingAttr();
  let paddingTensor: Tensor;
  if (!paddingAttr.pads.every((x) => x == 0)) {
    paddingTensor = await handlePadding(input, paddingAttr, device);
  } else {
    paddingTensor = input;
  }

  const outputSize = input.shape[0];
  const outputChannel = weight.shape[0];
  const maxY = paddingTensor.shape[2] - attr.kernelShape[0];
  const maxX = paddingTensor.shape[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    outputSize,
    outputChannel,
    outputHeight,
    outputWidth,
  ]);

  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: paddingTensor,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: paddingTensor.shape,
    },
    {
      rtype: RType.OutputTensor,
      data: output,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: output.shape,
    },
    {
      rtype: RType.InputTensor,
      data: weight,
    },
    {
      rtype: RType.InputTensor,
      data: bias ? bias : TensorBuilder.withShape([output.shape[1]]),
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [
        weight.shape[1],
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.strides[0],
        attr.strides[1],
      ],
    },
  ];
  const program: Program = {
    code: conv,
    entry: "conv",
  };
  const result = await computePass(
    resources,
    [
      Math.ceil(output.shape[3] / 8),
      Math.ceil(output.shape[2] / 8),
      Math.ceil(output.shape[1] / 4),
    ],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  output.data = result;

  return output;
}
