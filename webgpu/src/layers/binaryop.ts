import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";

export async function handleBinaryOp(a: Tensor, b: Tensor, binaryop: string, device: GPUDevice): Promise<Tensor> {
    const outputShape = getBroadcastShape(a.shape, b.shape);
    let output = TensorBuilder.withShape(outputShape);

    const aBroadcastDim = getBroadcastDims(a.shape, outputShape);
    const bBroadcastDim = getBroadcastDims(b.shape, outputShape);

    let gpuOutputBuffer;
    let computePipeline;
    let bindGroup;
    if (aBroadcastDim.length + bBroadcastDim.length == 0) {
        const gpuABuffer = a.setInputGPUBuffer(device);
        const gpuBBuffer = b.setInputGPUBuffer(device);
        gpuOutputBuffer = output.setOutputGPUBuffer(device);
        const gpuOutLen = setGPUReadBuffer(new Uint32Array([output.getLength()]), DType.uint32, device);

        const shaderModule = device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, write> output: array<f32>;
                @group(0) @binding(3) var<storage, read> len: u32; 
                
                let workgroup_size_x = 256;
                
                @stage(compute) 
                @workgroup_size(workgroup_size_x)
                fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                    for (var i: u32 = global_id.x; i < len; i += u32(workgroup_size_x)) {
                        output[i] = a[i] ${binaryop} b[i];
                    }
                }
            `
        });
        computePipeline = device.createComputePipeline({
            compute: {
                module: shaderModule,
                entryPoint: "main"
            }
        });

        bindGroup = createBindGroup(computePipeline, [
            gpuABuffer,
            gpuBBuffer,
            gpuOutputBuffer,
            gpuOutLen,
        ], device);
    } else {
        const gpuABuffer = a.setInputGPUBuffer(device);
        const aStrideVec4 = arrayToVec4(a.getStride()!);
        const aBroadcastDimVec4 = arrayToVec4(aBroadcastDim);
        const gpuAMetaBuffer = setGPUReadBuffer(new Uint32Array([
            a.ndim, 0, 0, 0,
            aStrideVec4[0], aStrideVec4[1], aStrideVec4[2], aStrideVec4[3],
            aBroadcastDimVec4[0], aBroadcastDimVec4[1], aBroadcastDimVec4[2], aBroadcastDimVec4[3]
        ]), DType.uint32, device);

        const gpuBBuffer = b.setInputGPUBuffer(device);
        const bStrideVec4 = arrayToVec4(b.getStride()!);
        const bBroadcastDimVec4 = arrayToVec4(bBroadcastDim);
        const gpuBMetaBuffer = setGPUReadBuffer(new Uint32Array([
            b.ndim, 0, 0, 0,
            bStrideVec4[0], bStrideVec4[1], bStrideVec4[2], bStrideVec4[3],
            bBroadcastDimVec4[0], bBroadcastDimVec4[1], bBroadcastDimVec4[2], bBroadcastDimVec4[3]
        ]), DType.uint32, device);

        gpuOutputBuffer = output.setOutputGPUBuffer(device);
        const outStrideVec4 = arrayToVec4(output.getStride()!);
        const gpuOutMetaBuffer = setGPUReadBuffer(new Uint32Array([
            output.getLength(), 0, 0, 0,
            outStrideVec4[0], outStrideVec4[1], outStrideVec4[2], outStrideVec4[3]
        ]), DType.uint32, device);
    
        const shaderModule = device.createShaderModule({
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
                
                fn idxToLoc(index: u32, stride: vec4<u32>) -> vec4<u32> {
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
                
                fn locToIdx(loc: vec4<u32>, stride: vec4<u32>) -> u32 {
                    var index: u32 = 0u;
                
                    for (var i: u32 = 0u; i < 4u; i++) {
                        index += loc[i] * stride[i];
                    }
                
                    return index;
                }
                
                fn getLoc(out_loc: vec4<u32>, dim: u32, broadcast_dim: vec4<u32>) -> vec4<u32> {
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
                    for (var i: u32 = global_id.x; i < out_meta.len; i += u32(workgroup_size_x)) {
                        let out_loc = idxToLoc(i, out_meta.stride);
                
                        let a_loc = getLoc(out_loc, a_meta.dim, a_meta.broadcast_dim);
                        let a_idx = locToIdx(a_loc, a_meta.stride);
                
                        let b_loc = getLoc(out_loc, b_meta.dim, b_meta.broadcast_dim);
                        let b_idx = locToIdx(b_loc, b_meta.stride);
                
                        out[i] = a[a_idx] ${binaryop} b[b_idx]; 
                    }
                }
            `
        });
        computePipeline = device.createComputePipeline({
            compute: {
                module: shaderModule,
                entryPoint: "main"
            }
        });
    
        bindGroup = createBindGroup(computePipeline, [
            gpuABuffer,
            gpuAMetaBuffer,
            gpuBBuffer,
            gpuBMetaBuffer,
            gpuOutputBuffer,
            gpuOutMetaBuffer
        ], device);
    }

    const commandEncoder = getCommandEncoder(computePipeline, bindGroup, [1], device);

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

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

function arrayToVec4(arr: number[]) {
    let vec4 = [0, 0, 0, 0];
    for (let i = 0; i < arr.length; i++) {
        vec4[3 - i] = arr[arr.length - 1 - i];
    }

    return vec4;
}