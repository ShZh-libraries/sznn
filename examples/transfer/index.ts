import { loadModel, TensorBuilder, Tensor } from "sznn-webgpu";
import { imageToArray, loadImage } from "../shared/cv";
import { displayImage, displayImageWithChannels } from "../shared/utils";

const WIDTH = 224;
const HEIGHT = 224;
const CHANNEL = WIDTH * HEIGHT;

function extractChannels(image: Tensor) {
  let red = [],
    blue = [],
    green = [];
  const bitmap = image.toArray();
  for (let i = 0; i < bitmap.length; i++) {
    if (i >= 2 * CHANNEL) {
      blue.push(bitmap[i]);
    } else if (i >= CHANNEL) {
      green.push(bitmap[i]);
    } else {
      red.push(bitmap[i]);
    }
  }

  return { red, green, blue };
}

let tensor;
let image;

const uploadBtn = document.querySelector("#upload-img") as HTMLButtonElement;
uploadBtn!.addEventListener("click", async () => {
  // Get file content by filesystem API
  let [fileHandle] = await (window as any).showOpenFilePicker();
  const file = await fileHandle.getFile();
  const content = await file.arrayBuffer();

  // Extract tensor from file content
  const buffer = Buffer.from(content);
  image = (await loadImage(buffer)).resize(HEIGHT, WIDTH); // Resize only
  const imageData = imageToArray(image);
  tensor = TensorBuilder.withAllArgs(imageData, [1, 3, HEIGHT, WIDTH]);

  // Display image
  const inputCanvas = document.querySelector(
    "#input-img"
  )! as HTMLCanvasElement;
  displayImage(inputCanvas, image);
  uploadBtn.style.display = "none";
});

const startBtn = document.querySelector("#start");
startBtn!.addEventListener("click", async () => {
  // Show spin loading
  const loading = document.querySelector("#loading") as HTMLDivElement;
  loading.style.display = "flex";

  // Load model and do inference
  const model = await loadModel("./model/mosaic.onnx");
  const result = await model.forward(tensor);
  const channels = extractChannels(result[0]);

  // Display image
  const outputCanvas = document.querySelector(
    "#output-img"
  )! as HTMLCanvasElement;
  displayImageWithChannels(outputCanvas, channels, WIDTH, HEIGHT);
  loading.style.display = "none";
});
