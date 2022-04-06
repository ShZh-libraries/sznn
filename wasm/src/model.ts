import { onnx } from "onnx-proto";
import { handle } from "./handler";
import { Tensor } from "./rs/pkg";
import { TensorDict } from "./tensor";
import { loadONNXModel } from "../../core/model";

export async function loadModel(path: string): Promise<Model> {
  const onnxModel = await loadONNXModel(path);
  const model = new Model(onnxModel);

  return model;
}

export class Model {
  dict: TensorDict;
  onnxModel: onnx.ModelProto;

  constructor(onnxModel: onnx.ModelProto) {
    this.onnxModel = onnxModel;
    this.dict = new TensorDict().init(onnxModel.graph!.initializer!);
  }

  forward(input: Tensor): Tensor[] {
    console.log("This is wasm!");

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

      if (Array.isArray(outputs)) {
        for (let outIndex = 0; outIndex < node.output!.length; outIndex++) {
          this.dict.set(node.output![outIndex], outputs[outIndex]);
        }
      } else {
        this.dict.set(node.output![0], outputs);
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
