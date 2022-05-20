import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";

const workgroup_size = 256;

export async function handleUnaryOp(
  input: Tensor,
  unaryop: string,
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
      rtype: RType.OutputTensor,
      data: output,
    },
    {
      rtype: RType.MetaUInt32Array,
      data: [len],
    },
  ];
  const program: Program = {
    code: `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, write> output: array<f32>;
            @group(0) @binding(2) var<storage, read> len: u32;

            let workgroup_size_x = ${workgroup_size};

            @stage(compute) 
            @workgroup_size(workgroup_size_x)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                if (global_id.x >= len) {
                    return;
                } 
                ${unaryop};
            }
        `,
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
