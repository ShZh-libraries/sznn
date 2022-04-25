/// <reference types="@webgpu/types" />

export { loadModel, Model } from "./model";
export { Tensor, TensorBuilder } from "./tensor";

// import { ConvAttr } from "../../core/attr/conv";
// import { getGPUDevice } from "./gpu";
// import { handleConv } from "./layers/conv";
// import { TensorBuilder } from "./tensor";

// export { handleAbs } from "./layers/abs";

// (async () => {
//   const device = await getGPUDevice();
//   const data = TensorBuilder.withData([
//     [
//       [
//         [0, 1, 2, 3, 4],
//         [5, 6, 7, 8, 9],
//         [10, 11, 12, 13, 14],
//         [15, 16, 17, 18, 19],
//         [20, 21, 22, 23, 24],
//         [25, 26, 27, 28, 29],
//         [30, 31, 32, 33, 34],
//       ],
//     ],
//   ]);
//   const weight = TensorBuilder.withData([
//     [
//       [
//         [1, 1, 1],
//         [1, 1, 1],
//         [1, 1, 1],
//       ],
//     ],
//   ]);

//   const attr = new ConvAttr();
//   attr.kernelShape = [3, 3];
//   attr.pads = [1, 0, 1, 0];
//   attr.strides = [2, 2];
//   const output = await handleConv(device!, attr, data, weight);
//   console.log(output);
// })()

// (async () => {
//   const device = await getGPUDevice();
//   const input = TensorBuilder.withData([-1, 0, 1]);

//   const output = await handleLeakyRelu(input, 0.1, device!);
//   console.log(output);
// })()
