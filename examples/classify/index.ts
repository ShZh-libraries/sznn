import Jimp from "jimp/*";
import { loadModel, TensorBuilder, Tensor } from "sznn-js";
import { imageToArray, loadImage, normalize } from "../shared/cv";
import { displayImage } from "../shared/utils";
import { imagenetClasses } from "./image_net";

const HEIGHT = 224;
const WIDTH = 224;

function preprocessing(image: Jimp): Tensor {
  const imageData = imageToArray(image);
  const normalized = normalize(imageData, 255);
  const tensor = TensorBuilder.withAllArgs(normalized, [1, 3, HEIGHT, WIDTH]);

  return tensor;
}

function postprocessing(output: Tensor): number {
  const outputArr = output.toArray();
  const maxProb = Math.max(...outputArr);
  const index = outputArr.indexOf(maxProb);

  return index;
}

// Main
const btn = document.querySelector("#btn");
btn!.addEventListener("click", async () => {
  // Get file content by filesystem API
  let [fileHandle] = await (window as any).showOpenFilePicker();
  const file = await fileHandle.getFile();
  const content = await file.arrayBuffer();

  // Transform file content to tensor and do preprocessing
  const buffer = Buffer.from(content);
  // Do resize here so we can display resized image later
  const image = (await loadImage(buffer)).resize(HEIGHT, WIDTH);
  const input = preprocessing(image);

  const canvas = document.querySelector("canvas")!;
  displayImage(canvas, image);

  // Load model and inference
  const model = await loadModel("./model/squeezenet.onnx");
  const output = model.forward(input)[0];
  const index = postprocessing(output);

  // Get and display result
  const clazz = imagenetClasses[index][1];
  const targetH = document.querySelector("h2")!;
  targetH.innerText = `"${clazz}"`;
});
