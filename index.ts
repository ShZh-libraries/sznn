import { AbsLayer } from "./js/layers/abs.js";
import { Tensor } from "./js/tensor.js";

const input = Tensor.withData([[1, -2], [-3, -4]]);
const output = Tensor.withShape([2, 2]);

console.log("Before");
for (let i = 0; i < 4; i++) {
    console.log(output.data[i]);
}

const layer = new AbsLayer();
layer.forward(input, output);

console.log("After");
for (let i = 0; i < 4; ++i) {
    console.log(output.data[i]);
}