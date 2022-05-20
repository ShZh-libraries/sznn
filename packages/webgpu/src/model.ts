import { onnx } from "onnx-proto";
import { loadONNXModel } from "../../common/model";
import { handle } from "./handler";
import { Tensor, TensorDict } from "./tensor";
import { ModelStat, caclAllTime } from "../../common/perf";

let startTime: number;
let endTime: number;
let stat: ModelStat = [];

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

  async forward(input: Tensor): Promise<Tensor[]> {
    // Put input tensor to tensor pool
    const inputName = this.onnxModel.graph!.node![0]!.input![0]!;
    this.dict.set(inputName, input);

    // Do forwarding
    for (const node of this.onnxModel.graph!.node!) {
      const inputs = node.input!.map((name) => this.dict.get(name)!)!;

      if (process.env.NODE_ENV !== "production") {
        startTime = performance.now();
      }

      const output = await handle(
        node.opType!,
        inputs,
        node.attribute! as onnx.AttributeProto[]
      );

      if (process.env.NODE_ENV !== "production") {
        endTime = performance.now();
        stat.push({
          op: node.opType!,
          name: node.name!,
          time: endTime - startTime,
        });
      }

      if (Array.isArray(output)) {
        for (let outIndex = 0; outIndex < node.output!.length; outIndex++) {
          this.dict.set(node.output![outIndex], output[outIndex]);
        }
      } else {
        this.dict.set(node.output![0], output);
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

    if (process.env.NODE_ENV !== "production") {
      console.log(stat);
      console.log(caclAllTime(stat));
    }

    return results;
  }
}
