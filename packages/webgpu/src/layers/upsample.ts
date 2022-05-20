import upsample from "./shaders/upsample.wgsl";
import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";
import { arrayToVec4 } from "../utils";

const workgroup_size = 256;

export async function handleUpSample(
  input: Tensor,
  scales: Tensor,
  device: GPUDevice
) {
  let outputShape = [];
  for (let shapeIndex = 0; shapeIndex < input.ndim; shapeIndex++) {
    outputShape.push(input.shape[shapeIndex] * scales.data[shapeIndex]);
  }
  let output = TensorBuilder.withShape(outputShape);

  const inputStrideVec4 = arrayToVec4(input.getStride()!);
  const outputStrideVec4 = arrayToVec4(output.getStride()!);
  let resouces: Resource[] = [
    {
      rtype: RType.InputTensor,
      data: input,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: inputStrideVec4,
    },
    {
      rtype: RType.InputTensor,
      data: scales,
    },
    {
      rtype: RType.OutputTensor,
      data: output,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: outputStrideVec4,
    },
  ];
  const program: Program = {
    code: upsample,
    entry: "main",
  };
  output.data = await computePass(
    resouces,
    [Math.ceil(output.getLength() / workgroup_size)],
    program,
    device,
    GPUDataEnum.Float32Array
  );

  return output;
}
