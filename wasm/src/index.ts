import * as Comlink from "comlink";
import { Tensor } from "./rs/pkg";

const { loadModel, TensorBuilder, Model } = Comlink.wrap(
    new Worker(new URL("./worker.ts", import.meta.url))
);

export { TensorBuilder, loadModel, Model, Tensor };