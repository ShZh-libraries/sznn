import padding from "./shaders/padding.wgsl";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";
import { PaddingAttr } from "../../../common/attr/padding";

export async function handlePadding(
  input: Tensor,
  attr: PaddingAttr,
  device: GPUDevice
): Promise<Tensor> {
  let outputShape = [];
  for (let index = 0; index < input.ndim; index++) {
    outputShape.push(
      attr.pads[index] + input.shape[index] + attr.pads[index + input.ndim]
    );
  }
  let output = TensorBuilder.withShape(outputShape);

  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: input.shape,
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
      rtype: RType.MetaUInt32Array,
      data: [attr.pads[2], attr.pads[3], attr.pads[6], attr.pads[7]],
    },
  ];
  const program: Program = {
    code: padding,
    entry: "main",
  };
  const result = await computePass(
    resources,
    [
      Math.ceil(output.shape[3] / 16),
      Math.ceil(output.shape[2] / 16),
      output.shape[1],
    ],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  output.data = result;

  return output;
}
