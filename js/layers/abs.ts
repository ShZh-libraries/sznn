import { Layer } from "../layer.js";
import { Tensor } from "../tensor.js";

export class AbsLayer extends Layer {
    forward(input: Tensor, output: Tensor) {
        for (let i = 0; i < input.data.length; i++) {
            output.data[i] = Math.abs(input.data[i]);
        }
    }
}
