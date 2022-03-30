import { onnx } from "onnx-proto";
import { handle } from "./handler";
import { Tensor, TensorDict } from "./tensor";

export async function loadModel(path: string): Promise<Model> {
  const onnxModel = await loadONNXModel(path);
  const model = new Model(onnxModel);

  return model;
}

export async function loadONNXModel(path: string): Promise<onnx.ModelProto> {
  const modelFile = await fetch(path);
  const modelFileContent = await modelFile.arrayBuffer();
  const model = onnx.ModelProto.decode(new Uint8Array(modelFileContent));

  return model;
}

export class Model {
  dict: TensorDict;
  onnxModel: onnx.ModelProto;

  constructor(onnxModel: onnx.ModelProto) {
    this.onnxModel = onnxModel;
    this.dict = new TensorDict().init(onnxModel.graph!.initializer!);
  }

  foward(input: Tensor): Tensor[] {
    // Put input tensor to tensor pool
    const inputName = this.onnxModel.graph!.node![0]!.input![0]!;
    this.dict.set(inputName, input);

    // Do forwarding
    for (const node of this.onnxModel.graph!.node!) {
      const inputs = node.input!.map((name) => this.dict.get(name)!)!;
      const outputs = handle(
        node.opType!,
        inputs,
        node.attribute! as onnx.AttributeProto[]
      );

      for (let outIndex = 0; outIndex < node.output!.length; outIndex++) {
        this.dict.set(node.output![outIndex], outputs[outIndex]);
      }
    }

    // Get result out of tensor pool
    const results: Tensor[] = [];
    for (const output of this.onnxModel.graph!.output!) {
      const result = this.dict.get(output.name!);
      if (result) {
        results.push(result);
      } else {
        throw new Error("The computation is not finished!");
      }
    }

    return results;
  }
}
