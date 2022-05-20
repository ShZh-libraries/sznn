import pooling from "./shaders/pooling.wgsl";
import { Tensor, TensorBuilder } from "../tensor";
import { PoolingAttr } from "../../../common/attr/pooling";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { handlePadding } from "./padding";

export async function handleMaxPool2D(
  input: Tensor,
  attr: PoolingAttr,
  device: GPUDevice
): Promise<Tensor> {
  // Padding first
  const paddingAttr = attr.getPaddingAttr();
  let paddingTensor: Tensor;
  if (!paddingAttr.pads.every((x) => x == 0)) {
    paddingTensor = await handlePadding(input, paddingAttr, device);
  } else {
    paddingTensor = input;
  }

  const maxY = paddingTensor.shape[2] - attr.kernelShape[0];
  const maxX = paddingTensor.shape[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    paddingTensor.shape[0],
    paddingTensor.shape[1],
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
      rtype: RType.MetaUInt32Array,
      data: [
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.strides[0],
        attr.strides[1],
      ],
    },
  ];
  const program: Program = {
    code: pooling,
    entry: "max_pool",
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

export async function handleAvgPool2D(
  input: Tensor,
  attr: PoolingAttr,
  device: GPUDevice
): Promise<Tensor> {
  // Padding first
  const paddingAttr = attr.getPaddingAttr();
  let paddingTensor: Tensor;
  if (!paddingAttr.pads.every((x) => x == 0)) {
    paddingTensor = await handlePadding(input, paddingAttr, device);
  } else {
    paddingTensor = input;
  }

  const maxY = paddingTensor.shape[2] - attr.kernelShape[0];
  const maxX = paddingTensor.shape[3] - attr.kernelShape[1];
  const outputHeight = Math.floor(maxY / attr.strides[0]) + 1;
  const outputWidth = Math.floor(maxX / attr.strides[1]) + 1;
  const output = TensorBuilder.withShape([
    paddingTensor.shape[0],
    paddingTensor.shape[1],
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
      rtype: RType.MetaUInt32Array,
      data: [
        attr.kernelShape[0],
        attr.kernelShape[1],
        attr.strides[0],
        attr.strides[1],
      ],
    },
  ];
  const program: Program = {
    code: pooling,
    entry: "avg_pool",
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

export async function handleGlobalAvgPool(
  input: Tensor,
  device: GPUDevice
): Promise<Tensor> {
  const globalAvgPoolingAttr = new PoolingAttr();
  globalAvgPoolingAttr.kernelShape = [input.shape[2], input.shape[3]];
  const output = await handleAvgPool2D(input, globalAvgPoolingAttr, device);

  return output;
}
