import { Tensor } from "./tensor.js";

export abstract class Layer {
    abstract forward(input: Tensor, output: Tensor): void; 
}