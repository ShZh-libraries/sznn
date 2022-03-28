import {onnx} from "onnx-proto";
import { handleBatchNorm } from "./layers/batchnorm";
import { handleConcat } from "./layers/concat";
import { handleConv } from "./layers/conv";
import { handleDropout } from "./layers/dropout";
import { handleAvgPool2D, handleGlobalAvgPool, handleMaxPool2D } from "./layers/pooling";
import { handleLeakyRelu, handleRelu } from "./layers/relu";
import { handleReshape } from "./layers/reshape";
import { handleSoftmax } from "./layers/softmax";
import { handleUpSample } from "./layers/upsample";
import { Tensor } from "./tensor";
import { TensorPool } from "./tensor_pool";

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
    pool: TensorPool;
    onnxModel: onnx.ModelProto;

    constructor(onnxModel: onnx.ModelProto) {
        this.onnxModel = onnxModel;
        this.pool = new TensorPool();

        // Initilaize tensor pool
        for (const initializer of onnxModel.graph?.initializer!) {
            this.pool.setFromInitializer(initializer as onnx.TensorProto);
        }

        console.log(this.pool);
    }

    foward(input: Tensor): Tensor[] {
        // Put input tensor to tensor pool
        const inputName = this.onnxModel.graph?.node?.[0]?.input?.[0]!;
        this.pool.set(inputName, input);

        // Do forwarding
        for (const node of this.onnxModel.graph?.node!) {
            const inputTensors = node.input?.map(name => this.pool.get(name)!)!;
            let outputs: Tensor[];
            switch (node.opType) {
                case "Conv":
                    outputs = handleConv(inputTensors, node.attribute! as onnx.AttributeProto[]); 
                    break;
                case "BatchNormalization":
                    outputs = handleBatchNorm(inputTensors);
                    break;
                case "Dropout":
                    outputs = handleDropout(inputTensors, node.attribute! as onnx.AttributeProto[]);
                    break;
                case "Relu":
                    outputs = handleRelu(inputTensors);
                    break;
                case "LeakyRelu":
                    outputs = handleLeakyRelu(inputTensors, node.attribute! as onnx.AttributeProto[]);
                    break;
                case "MaxPool":
                    outputs = handleMaxPool2D(inputTensors, node.attribute! as onnx.AttributeProto[]);
                    break;
                case "AveragePool":
                    outputs = handleAvgPool2D(inputTensors, node.attribute! as onnx.AttributeProto[]);
                    break;
                case "GlobalAveragePool":
                    outputs = handleGlobalAvgPool(inputTensors);
                    break;
                case "UpSample":
                    outputs = handleUpSample(inputTensors);
                    break;
                case "Concat":
                    outputs = handleConcat(inputTensors, node.attribute! as onnx.AttributeProto[]);
                    break;
                case "Softmax":
                    outputs = handleSoftmax(inputTensors);
                    break;
                case "Reshape":
                    outputs = handleReshape(inputTensors);
                    break;
                default:
                    throw new Error(`Unknown op type ${node.opType}! The node name is ${node.name}`);
            }

            console.log(node.opType, outputs);
            for (let outIndex = 0; outIndex < node.output!.length; outIndex++) {
                this.pool.set(node.output![outIndex], outputs[outIndex]);
            }
        }

        // Get result out of tensor pool
        const results: Tensor[] = [];
        for (const output of this.onnxModel.graph?.output!) {
            const result = this.pool.get(output.name!);
            if (result) {
                results.push(result);
            } else {
                throw new Error("The computation is not finished!");
            }
        }

        return results;
    }
}