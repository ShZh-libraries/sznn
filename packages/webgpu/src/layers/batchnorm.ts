import batchnorm from "./shaders/batchnorm.wgsl";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";

const workgroup_size = 256;

export async function handleBatchNorm(
  input: Tensor,
  scale: Tensor,
  bias: Tensor,
  mean: Tensor,
  variance: Tensor,
  device: GPUDevice
): Promise<Tensor> {
  let output = TensorBuilder.withShape(input.shape);

  const len = output.getLength();
  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.InputTensor,
      data: scale,
    },
    {
      rtype: RType.InputTensor,
      data: bias,
    },
    {
      rtype: RType.InputTensor,
      data: mean,
    },
    {
      rtype: RType.InputTensor,
      data: variance,
    },
    {
      rtype: RType.OutputTensor,
      data: output,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [len],
    },
  ];
  const program: Program = {
    code: batchnorm,
    entry: "main",
  };

  const result = await computePass(
    resources,
    [Math.ceil(len / workgroup_size)],
    program,
    device,
    GPUDataEnum.Float32Array
  );
  output.data = result;

  return output;
}
