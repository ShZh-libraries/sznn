/// <reference types="@webgpu/types" />

import { getGPUDevice } from "./gpu";
import { handleAbs } from "./layers/abs";
import { TensorBuilder } from "./tensor";

export { handleAbs } from "./layers/abs";

(async () => {
  const device = await getGPUDevice();
  const input = TensorBuilder.withData([[-1, 0, 1]]);
  const output = await handleAbs(input, device!);
  console.log(output);
})()
