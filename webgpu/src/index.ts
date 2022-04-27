/// <reference types="@webgpu/types" />

export { loadModel, Model } from "./model";
export { Tensor, TensorBuilder } from "./tensor";

// import { ConvAttr } from "../../core/attr/conv";
// import { getGPUDevice } from "./gpu";
// import { handleConv } from "./layers/conv";
// import { TensorBuilder } from "./tensor";

// (async () => {
//   const device = await getGPUDevice();
//   const input = TensorBuilder.withData([-1, 0, 1]);

//   const output = await handleLeakyRelu(input, 0.1, device!);
//   console.log(output);
// })()
