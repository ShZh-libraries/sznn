import mean from "./shaders/instancenorm/mean.wgsl";
import variance from "./shaders/instancenorm/variance.wgsl";
import instancenorm from "./shaders/instancenorm/index.wgsl";
import { Tensor, TensorBuilder } from "../tensor";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";

async function caclMean(input: Tensor, device: GPUDevice) {
  const channelSize = input.shape[2] * input.shape[3];
  let means = TensorBuilder.withShape([input.shape[0], input.shape[1]]);
  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.OutputTensor,
      data: means,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [input.shape[1]],
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [channelSize],
    },
  ];
  const program: Program = {
    code: mean,
    entry: "main",
  };
  means.data = await computePass(
    resources,
    [Math.ceil(input.shape[1] / 8), Math.ceil(channelSize / 32)],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  return means;
}

async function calcVariance(input: Tensor, means: Tensor, device: GPUDevice) {
  const channelSize = input.shape[2] * input.shape[3];
  let variances = TensorBuilder.withShape([input.shape[0], input.shape[1]]);
  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.InputTensor,
      data: means,
    },
    {
      rtype: RType.OutputTensor,
      data: variances,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [input.shape[1]],
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [channelSize],
    },
  ];
  const program: Program = {
    code: variance,
    entry: "main",
  };
  variances.data = await computePass(
    resources,
    [Math.ceil(input.shape[1] / 8), Math.ceil(channelSize / 32)],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  return variances;
}

async function forward(
  input: Tensor,
  weight: Tensor,
  bias: Tensor,
  means: Tensor,
  variances: Tensor,
  epsilon: number,
  device: GPUDevice
) {
  let output = TensorBuilder.withShape(input.shape);
  const len = output.getLength();
  const channelSize = input.shape[2] * input.shape[3];
  let resources: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.InputTensor,
      data: weight,
    },
    {
      rtype: RType.InputTensor,
      data: bias,
    },
    {
      rtype: RType.InputTensor,
      data: means,
    },
    {
      rtype: RType.InputTensor,
      data: variances,
    },
    {
      rtype: RType.MetaFloat32Array,
      data: [epsilon],
    },
    {
      rtype: RType.OutputTensor,
      data: output,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [len, channelSize],
    },
  ];
  const program: Program = {
    code: instancenorm,
    entry: "main",
  };
  output.data = await computePass(
    resources,
    [Math.ceil(len / 256)],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  return output;
}

export async function handleInstanceNorm(
  input: Tensor,
  weight: Tensor,
  bias: Tensor,
  epsilon: number,
  device: GPUDevice
): Promise<Tensor> {
  const means = await caclMean(input, device);
  const variances = await calcVariance(input, means, device);
  const output = await forward(
    input,
    weight,
    bias,
    means,
    variances,
    epsilon,
    device
  );

  return output;
}
