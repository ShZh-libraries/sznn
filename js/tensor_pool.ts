import { onnx } from "onnx-proto";
import { Tensor, TensorBuilder } from "./tensor";

export class TensorPool {
    private pool: Map<string, Tensor> = new Map();

    get(name: string): Tensor | undefined {
        return this.pool.get(name);
    }

    set(name: string, tensor: Tensor): void {
        this.pool.set(name, tensor);
    }

    setFromInitializer(initializer: onnx.TensorProto): void {
        const tensor = TensorBuilder.withInitializer(initializer);
        this.pool.set(initializer.name, tensor);
    }
}