import init, { initThreadPool } from "./rs/pkg";
import { loadModel, Model } from "./model";
import { TensorBuilder as WasmBuilder } from "./tensor";

import * as Comlink from "comlink";

(async () => {
  await init();
  await initThreadPool(navigator.hardwareConcurrency);
})();

Comlink.expose({
  loadModel,
  withAllArgs: WasmBuilder.withAllArgs,
  Model: Comlink.proxy(Model),
});
