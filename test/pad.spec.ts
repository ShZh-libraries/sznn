import { forwardConstantPad, PaddingAttr } from "../js/layers/pad";
import { TensorBuilder } from "../js/tensor"

describe("Test JS backend for padding layer", () => {
    test("Test constant padding", () => {
        const input = TensorBuilder.withData([
            [1, 1.2], [2.3, 3.4], [4.5, 5.7]
        ]);
        const attr = new PaddingAttr();
        attr.pads = [0, 2, 0, 0];

        const output = forwardConstantPad(input, attr);
        expect(output.shape).toEqual([3, 4]);
        expect(output.data).toEqual(new Float32Array([
            0, 0, 1, 1.2, 0, 0, 2.3, 3.4, 0, 0, 4.5, 5.7
        ]));
    })
})