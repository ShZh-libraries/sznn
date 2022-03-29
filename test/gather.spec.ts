import { forward } from "../js/layers/gather";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backends for gather layer", () => {
    test("Test simple gather", () => {
        const input = TensorBuilder.withData([[
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]]);
        const indices = TensorBuilder.withData([1]);
        const output = forward(input, indices);

        expect(output.shape).toEqual([1, 1, 2, 2]);
        expect(output.data).toEqual(new Float32Array([5, 6, 7, 8]));
    });
});