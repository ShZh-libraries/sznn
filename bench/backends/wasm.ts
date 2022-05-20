import Jimp from "jimp/*";
import { loadModel, TensorBuilder, Tensor } from "sznn-wasm";
import { imageToArray, normalize } from "../utils/cv";

async function preprocessing(image: Jimp, height: number, width: number) {
  const imageData = imageToArray(image);
  const normalized = normalize(imageData, 255);
  const tensor = await TensorBuilder.withAllArgs(normalized, [
    1,
    3,
    height,
    width,
  ]);

  return tensor;
}

async function postprocessing(output: Tensor) {
  const outputArr = await output.toArray();
  const maxProb = Math.max(...outputArr);
  const index = outputArr.indexOf(maxProb);

  return index;
}

export async function inferenceWASM(
  image: Jimp,
  height: number,
  width: number
) {
  const input = await preprocessing(image, height, width);
  const model = await loadModel("../model/squeezenet.onnx");
  const start = performance.now();
  const output = (await model.forward(input))[0];
  const elapse = Math.floor(performance.now() - start);
  const index = await postprocessing(output);

  return { elapse, index };
}
