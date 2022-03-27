import { forward, MaxPoolAttr } from "../js/layers/pooling";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backend for max pooling layer", () => {
    test("Test normal max pooling layer", () => {
        const src = TensorBuilder.withData([[
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]
            ]
        ]]);
        const maxPoolingAttr = new MaxPoolAttr();
        maxPoolingAttr.kernelShape = [2, 2];
        maxPoolingAttr.strides = [1, 1];

        const result = forward(src, maxPoolingAttr);

        expect(result.shape).toEqual([1, 2, 2, 2]);
        expect(result.data).toEqual(new Float32Array([
            5, 6, 8, 9, 14, 15, 17, 18
        ]))
    });
});