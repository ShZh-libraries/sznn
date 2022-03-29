import { forward } from "../js/layers/floor";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backends for floor layer", () => {
    test("Test floor layer", () => {
        const src = TensorBuilder.withData([-1.5, 1.2, 2]);
        const result = forward(src);

        expect(result.shape).toEqual([3]);
        expect(result.data).toEqual(new Float32Array([-2, 1, 2]));
    });
});