import { Tensor } from "./tensor.js";

export abstract class Layer {
    loadParams?(): void;
    
    loadWeights?(): void;

    abstract forward(input: Tensor, output: Tensor): void; 
}