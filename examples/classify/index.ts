import { loadModel } from "../../js/model";
import { imageToTensor, loadImage } from "../../utils/cv";
import { imagenetClasses } from "./image_net";

const HEIGHT = 224;
const WIDTH = 224;

const btn = document.querySelector("#btn");
btn!.addEventListener("click", async () => {
  // Get file content by filesystem API
  let [fileHandle] = await (window as any).showOpenFilePicker();
  const file = await fileHandle.getFile();
  const content = await file.arrayBuffer();

  // Transform file content to tensor
  const buffer = Buffer.from(content);
  const image = (await loadImage(buffer)).resize(HEIGHT, WIDTH);
  const tensor = imageToTensor(image).normalize(255);

  // Display image
  const canvas = document.querySelector("canvas")!;
  const context = canvas.getContext("2d")!;
  const imageData = context.createImageData(HEIGHT, WIDTH);
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i + 0] = image.bitmap.data[i + 0]; // R value
    imageData.data[i + 1] = image.bitmap.data[i + 1]; // G value
    imageData.data[i + 2] = image.bitmap.data[i + 2]; // B value
    imageData.data[i + 3] = image.bitmap.data[i + 3]; // A value
  }
  context.putImageData(imageData, 0, 0);

  // Load model and inference
  const model = await loadModel("./model/squeezenet.onnx");
  const result = model.forward(tensor)[0];
  let maxIndex = 0;
  let max = result.data[0];
  for (let index = 0; index < result.data.length; index++) {
    if (max < result.data[index]) {
      max = result.data[index];
      maxIndex = index;
    }
  }

  // Get and display result
  const clazz = imagenetClasses[maxIndex][1];
  const targetH = document.querySelector("h2")!;
  targetH.innerText = `"${clazz}"`;
});
