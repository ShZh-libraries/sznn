import { onnx } from "onnx-proto";
import { handle } from "./handler";
import { Tensor } from "./rs/pkg";
import { TensorDict } from "./tensor";
import { loadONNXModel } from "../../common/model";
import { caclAllTime, ModelStat } from "../../common/perf";
import * as Comlink from "comlink";

let startTime: number;
let endTime: number;
let stat: ModelStat = [];

export async function loadModel(path: string) {
  const onnxModel = await loadONNXModel(path);
  const model = new Model(onnxModel);

  return Comlink.proxy(model);
}

export class Model {
  dict: TensorDict;
  onnxModel: onnx.ModelProto;

  constructor(onnxModel: onnx.ModelProto) {
    this.onnxModel = onnxModel;
    this.dict = new TensorDict().init(onnxModel.graph!.initializer!);
  }

  forward(inputPtr: Tensor): Tensor[] {
    console.log("This is wasm!");
    let input = new Tensor();
    input.free(); // Avoid memory leakage
    Object.assign(input, inputPtr);

    // Put input tensor to tensor pool
    const inputName = this.onnxModel.graph!.node![0]!.input![0]!;
    this.dict.set(inputName, input);

    // Do forwarding
    for (const node of this.onnxModel.graph!.node!) {
      const inputs = node.input!.map((name) => this.dict.get(name)!)!;

      if (process.env.NODE_ENV !== "production") {
        startTime = performance.now();
      }

      const outputs = handle(
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

    if (process.env.NODE_ENV !== "production") {
      console.log(stat);
      console.log(caclAllTime(stat));
    }

    return Comlink.proxy(results);
  }
}
