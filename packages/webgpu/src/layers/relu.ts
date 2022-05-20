import {
  computePass,
  GPUDataEnum,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";

const workgroup_size = 256;

export async function handleRelu(
  input: Tensor,
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
            
            let workgroup_size_x = 256;
            
            @stage(compute) 
            @workgroup_size(workgroup_size_x)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                if (global_id.x >= len) {
                    return;
                }
                output[global_id.x] = max(input[global_id.x], 0.);
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

export async function handleLeakyRelu(
  input: Tensor,
  alpha: number,
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
    {
      rtype: RType.MetaFloat32Array,
      data: [alpha],
    },
  ];
  const program: Program = {
    code: `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, write> output: array<f32>;
            @group(0) @binding(2) var<storage, read> len: u32;
            @group(0) @binding(3) var<storage, read> alpha: f32;
            
            let workgroup_size_x = 256;
            
            @stage(compute) 
            @workgroup_size(workgroup_size_x)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                if (global_id.x >= len) {
                    return;
                }
                output[global_id.x] = max(input[global_id.x], 0.) * alpha;
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
