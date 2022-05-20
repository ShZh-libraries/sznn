import {
  computePass,
  GPUDataEnum,
  GPUDataType,
  Program,
  Resource,
  ResourceType as RType,
} from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";
import { arrayToVec4 } from "../utils";

const workgroup_size = 256;

export async function handleBinaryOp(
  a: Tensor,
  b: Tensor,
  binaryop: string,
  device: GPUDevice
): Promise<Tensor> {
  const outputShape = getBroadcastShape(a.shape, b.shape);
  let output = TensorBuilder.withShape(outputShape);
  const len = output.getLength();

  const aBroadcastDim = getBroadcastDims(a.shape, outputShape);
  const bBroadcastDim = getBroadcastDims(b.shape, outputShape);

  let result: GPUDataType;
  if (aBroadcastDim.length + bBroadcastDim.length == 0) {
    let resources: Resource[] = [
      {
        rtype: RType.InputTensor,
        data: a,
      },
      {
        rtype: RType.InputTensor,
        data: b,
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
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, write> output: array<f32>;
                @group(0) @binding(3) var<storage, read> len: u32; 
                
                let workgroup_size_x = ${workgroup_size};
                
                @stage(compute) 
                @workgroup_size(workgroup_size_x)
                fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                    if (global_id.x >= len) {
                        return;
                    }
                    output[global_id.x] = a[global_id.x] ${binaryop} b[global_id.x];
                }
            `,
      entry: "main",
    };

    result = await computePass(
      resources,
      [Math.ceil(len / workgroup_size)],
      program,
      device,
      GPUDataEnum.Float32Array
    );
  } else {
    const len = output.getLength();
    const aStrideVec4 = arrayToVec4(a.getStride()!);
    const aBroadcastDimVec4 = arrayToVec4(aBroadcastDim);
    const bStrideVec4 = arrayToVec4(b.getStride()!);
    const bBroadcastDimVec4 = arrayToVec4(bBroadcastDim);
    const outStrideVec4 = arrayToVec4(output.getStride()!);

    const resources: Resource[] = [
      {
        rtype: RType.InputTensor,
        data: a,
      },
      {
        rtype: RType.MetaUInt32Array,
        data: [a.ndim, 0, 0, 0, ...aStrideVec4, ...aBroadcastDimVec4],
      },
      {
        rtype: RType.InputTensor,
        data: b,
      },
      {
        rtype: RType.MetaUInt32Array,
        data: [b.ndim, 0, 0, 0, ...bStrideVec4, ...bBroadcastDimVec4],
      },
      {
        rtype: RType.OutputTensor,
        data: output,
      },
      {
        rtype: RType.MetaUInt32Array,
        data: [len, 0, 0, 0, ...outStrideVec4],
      },
    ];
    const program: Program = {
      code: `
                struct InTensorMeta {
                    dim: u32,
                    stride: vec4<u32>,
                    broadcast_dim: vec4<u32>,
                }
                
                struct OutTensorMeta {
                    len: u32,
                    stride: vec4<u32>,
                }
                
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> a_meta: InTensorMeta;
                @group(0) @binding(2) var<storage, read> b: array<f32>;
                @group(0) @binding(3) var<storage, read> b_meta: InTensorMeta;
                @group(0) @binding(4) var<storage, write> out: array<f32>;
                @group(0) @binding(5) var<storage, read> out_meta: OutTensorMeta;
                
                let workgroup_size_x = 256;
                
                fn idx_to_loc(index: u32, stride: vec4<u32>) -> vec4<u32> {
                    var loc = vec4<u32>();
                    var idx = index;
                
                    for (var i: u32 = 0u; i < 4u; i++) {
                        if (stride[i] != 0u) {
                            loc[i] = idx / stride[i];
                            idx %= stride[i];
                        }
                    }
                
                    return loc;
                }
                
                fn loc_to_idx(loc: vec4<u32>, stride: vec4<u32>) -> u32 {
                    var index: u32 = 0u;
                
                    for (var i: u32 = 0u; i < 4u; i++) {
                        index += loc[i] * stride[i];
                    }
                
                    return index;
                }
                
                fn get_loc(out_loc: vec4<u32>, dim: u32, broadcast_dim: vec4<u32>) -> vec4<u32> {
                    var loc = vec4<u32>();
                
                    for (var i: u32 = 4u - dim; i < 4u; i++) {
                        if (broadcast_dim[0] != i && broadcast_dim[1] != i && broadcast_dim[2] != i && broadcast_dim[3] != i) {
                            loc[i] = out_loc[i];
                        }
                    }
                
                    return loc;
                } 
                
                
                @stage(compute) 
                @workgroup_size(workgroup_size_x)
                fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                    if (global_id.x >= out_meta.len) {
                        return;
                    }

                    let out_loc = idx_to_loc(global_id.x, out_meta.stride);
                
                    let a_loc = get_loc(out_loc, a_meta.dim, a_meta.broadcast_dim);
                    let a_idx = loc_to_idx(a_loc, a_meta.stride);
            
                    let b_loc = get_loc(out_loc, b_meta.dim, b_meta.broadcast_dim);
                    let b_idx = loc_to_idx(b_loc, b_meta.stride);
            
                    out[global_id.x] = a[a_idx] ${binaryop} b[b_idx]; 
                }
            `,
      entry: "main",
    };
    result = await computePass(
      resources,
      [Math.ceil(len / workgroup_size)],
      program,
      device,
      GPUDataEnum.Float32Array
    );
  }

  output.data = result!;

  return output;
}

function getBroadcastShape(shape1: number[], shape2: number[]): number[] {
  let result = [];
  let resultLength =
    shape1.length > shape2.length ? shape1.length : shape2.length;

  for (let index = 0; index < resultLength; index++) {
    let a = shape1[shape1.length - 1 - index]
      ? shape1[shape1.length - 1 - index]
      : 1;
    let b = shape2[shape2.length - 1 - index]
      ? shape2[shape2.length - 1 - index]
      : 1;

    if (a == 1) {
      result.unshift(b);
    } else if (b == 1) {
      result.unshift(a);
    } else if (a == b) {
      result.unshift(a);
    } else {
      throw new Error("Cannot broadcast!!");
    }
  }

  return result;
}

function getBroadcastDims(shape: number[], resultShape: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    const dim = shape.length - 1 - i;
    const a = shape[dim] || 1;
    const b = resultShape[dim] || 1;
    if (b > 1 && a === 1) {
      result.unshift(dim);
    }
  }
  return result;
}
