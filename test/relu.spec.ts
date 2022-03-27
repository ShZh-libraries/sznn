import { forward } from "../js/layers/relu";
import { Tensor, TensorBuilder } from "../js/tensor";

describe("Test JS backend for leaky relu layer", () => {
    test("Test leaky relu correctness", () => {
        const input = TensorBuilder.withData([-1, 0, 1]);
        const alpha = 0.5;

        const result = forward(input, alpha);
        expect(result.shape).toEqual([3]);
        expect(result.data).toEqual(new Float32Array([-0.5, 0, 1]));
    })
});