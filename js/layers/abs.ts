import { Tensor } from "../tensor.js";

export function inplace_abs(input: Tensor) {
    for (let i = 0; i < input.data.length; i++) {
        input.data[i] = Math.abs(input.data[i]);
    }    
}