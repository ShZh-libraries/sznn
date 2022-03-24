import { Tensor } from "./js/tensor.js";

const tensor = new Tensor([[1, 2], [3, 4]], [2, 2]);
for (let i = 0; i < 4; ++i) {
    console.log(tensor.data[i]);
}