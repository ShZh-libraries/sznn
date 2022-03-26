import {onnx} from "onnx-proto";

export async function loadONNXModel(path: string): Promise<onnx.ModelProto> {
    const modelFile = await fetch(path);
    const modelFileContent = await modelFile.arrayBuffer();
    const model = onnx.ModelProto.decode(new Uint8Array(modelFileContent));

    return model;
}

export function modelTransformation(onnxModel: onnx.ModelProto) {
    const test = onnxModel.graph?.initializer![0] as onnx.TensorProto;
    return test;
}