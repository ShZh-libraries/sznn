import { Tensor } from "./js/tensor.js";
import { inplace_abs } from "./js/layers/abs.js";

const tensor = new Tensor([[1, -2], [-3, -4]], [2, 2]);

inplace_abs(tensor);

for (let i = 0; i < 4; ++i) {
    console.log(tensor.data[i]);
}