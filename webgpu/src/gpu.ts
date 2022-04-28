import { DType } from "./tensor";

type GPUDataType =
    | Int32Array
    | Uint32Array
    | Float32Array;

export async function getGPUDevice() {
    if (!("gpu" in navigator)) {
        console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.log("Failed to get GPU adapter.");
        return;
    }
    return await adapter.requestDevice();
}

export function setGPUReadBuffer(data: GPUDataType, dtype: DType, device: GPUDevice): GPUBuffer {
    const gpuBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE
    });

    const arrayBuffer = gpuBuffer.getMappedRange();
    switch (dtype) {
        case DType.int32:
            new Int32Array(arrayBuffer).set(data);
            break;
        case DType.uint32:
            new Uint32Array(arrayBuffer).set(data);
            break;
        case DType.float32:
            new Float32Array(arrayBuffer).set(data);
            break;
        default:
            throw new Error("Type not support in Web GPU!");
    }

    gpuBuffer.unmap();

    return gpuBuffer;
}

export function loadWGSL(code: any, device: GPUDevice, entryPoint: string = "main"): GPUComputePipeline {
    const shaderModule = device.createShaderModule({ code });
    const computePipeline = device.createComputePipeline({
        compute: {
            module: shaderModule,
            entryPoint
        }
    });

    return computePipeline;
}

export function createBindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[], device: GPUDevice): GPUBindGroup {
    let entries = [];
    for (let i = 0; i < buffers.length; i++) {
        const entry = {
            binding: i,
            resource: {
                buffer: buffers[i]
            }
        };
        entries.push(entry);
    }

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries
    });

    return bindGroup;
}

export function getCommandEncoder(pipeline: GPUComputePipeline, group: GPUBindGroup, dispatch: number[], device: GPUDevice): GPUCommandEncoder {
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, group);
    passEncoder.dispatch(dispatch[0], dispatch[1], dispatch[2]);
    passEncoder.end();

    return commandEncoder;
}

export async function getResult(encoder: GPUCommandEncoder, outputBuffer: GPUBuffer, size: number, device: GPUDevice): Promise<ArrayBuffer> {
    const gpuReadBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    encoder.copyBufferToBuffer(outputBuffer, 0, gpuReadBuffer, 0, size);

    // Submit GPU commands.
    const gpuCommands = encoder.finish();
    device.queue.submit([gpuCommands]);

    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const resultBuffer = gpuReadBuffer.getMappedRange();
    
    return resultBuffer;
}