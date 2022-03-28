import { forwardRelu, forwardLeakRelu} from "../js/layers/relu";
import { TensorBuilder } from "../js/tensor";

describe("Test JS backend for relu layer", () => {
    test("Test relu", () => {
        const input = TensorBuilder.withData([-1, 0, 1]);

        const result = forwardRelu(input);
        expect(result.shape).toEqual([3]);
        expect(result.data).toEqual(new Float32Array([0, 0, 1]));
    });

    test("Test leaky relu", () => {
        const input = TensorBuilder.withData([-1, 0, 1]);
        const alpha = 0.5;

        const result = forwardLeakRelu(input, alpha);
        expect(result.shape).toEqual([3]);
        expect(result.data).toEqual(new Float32Array([-0.5, 0, 1]));
    });
});