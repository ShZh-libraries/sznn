import init, { initThreadPool } from "./rs/pkg";
import { loadModel, Model } from "./model";
import { TensorBuilder } from "./tensor";

import * as Comlink from "comlink";

(async () => {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
})()

Comlink.expose({
    loadModel,
    TensorBuilder: TensorBuilder.withAllArgs,
    Model: Comlink.proxy(Model)
})


