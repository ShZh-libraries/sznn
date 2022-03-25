import { Layer } from "./js/layer.js";
import { AbsLayer } from "./js/layers/abs.js";
import { Tensor } from "./js/tensor.js";

let input = Tensor.withData([[1, -2], [-3, -4]]);
let output = Tensor.withShape([2, 2]);

console.log("Before");
for (let i = 0; i < 4; i++) {
    console.log(output.data[i]);
}

let layers: Layer[] = [new AbsLayer()];
for (let layer of layers) {
    layer.forward(input, output);
    input = output;
}

let result = input;
console.log("After");
for (let i = 0; i < 4; ++i) {
    console.log(result.data[i]);
}