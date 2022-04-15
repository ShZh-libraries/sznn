import { initThreadPool } from "./rs/pkg/index";

(async () => {
    await initThreadPool(navigator.hardwareConcurrency);
})()

export { Tensor } from "./rs/pkg";
export { loadModel, Model } from "./model";
