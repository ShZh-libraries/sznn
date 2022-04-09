import padding from "./wgsl/padding.wgsl";
import { createBindGroup, getCommandEncoder, getResult, loadWGSL, setGPUReadBuffer } from "../gpu";
import { DType, Tensor, TensorBuilder } from "../tensor";
import { PaddingAttr } from "../../../core/attr/padding";

export async function handlePadding(input: Tensor, attr: PaddingAttr, device: GPUDevice): Promise<Tensor> {
    let out_shape = [
        input.shape[0], input.shape[1], 
        input.shape[2] + attr.pads[0] + attr.pads[2],
        input.shape[3] + attr.pads[1] + attr.pads[3]
    ];
    let output = TensorBuilder.withShape(out_shape);

    const gpuInputBuffer = input.setInputGPUBuffer(device);
    const gpuInShapeBuffer = setGPUReadBuffer(new Uint32Array(input.shape), DType.uint32, device);
    const gpuOutputBuffer = output.setOutputGPUBuffer(device);
    const gpuOutShapeBuffer = setGPUReadBuffer(new Uint32Array(output.shape), DType.uint32, device);
    const gpuAttrBuffer = setGPUReadBuffer(new Uint32Array([attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3]]), DType.uint32, device);

    const computePipeline = loadWGSL(padding, device);

    const bindGroup = createBindGroup(computePipeline, [
        gpuInputBuffer, gpuInShapeBuffer,
        gpuOutputBuffer, gpuOutShapeBuffer,
        gpuAttrBuffer
    ], device);

    const commandEncoder = getCommandEncoder(
        computePipeline, 
        bindGroup, 
        [Math.ceil(output.shape[3] / 16), Math.ceil(output.shape[2] / 16), output.shape[1]], 
        device
    );

    const resultBuffer = await getResult(commandEncoder, gpuOutputBuffer, output.data.byteLength, device);
    const resultArray = new Float32Array(resultBuffer);

    output.data = resultArray;

    return output;
}

