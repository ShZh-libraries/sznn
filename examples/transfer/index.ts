import { loadModel } from "../../js/model";
import { Tensor } from "../../js/tensor";
import { Image, imageToTensor, loadImage } from "../../utils/image";

const WIDTH = 224;
const HEIGHT = 224;
const CHANNEL = WIDTH * HEIGHT;

let image: Image;
let tensor: Tensor;

const uploadBtn = document.querySelector("#upload-img");
uploadBtn!.addEventListener("click", async () => {
    // Get file content by filesystem API
    let [fileHandle] = await (window as any).showOpenFilePicker();
    const file = await fileHandle.getFile();
    const content = await file.arrayBuffer();

    // Cast file content to tensor
    const buffer = Buffer.from(content);
    image = (await loadImage(buffer)).resize(HEIGHT, WIDTH);
    tensor = imageToTensor(image);

    // Display image
    const canvasInput = document.querySelector("#input-img")! as HTMLCanvasElement;
    const contextInput = canvasInput.getContext("2d")!;
    const inputImageData = contextInput.createImageData(HEIGHT, WIDTH);
    for (let i = 0; i < inputImageData.data.length; i += 4) {
        inputImageData.data[i + 0] = image.bitmap.data[i + 0];  // R value
        inputImageData.data[i + 1] = image.bitmap.data[i + 1];  // G value
        inputImageData.data[i + 2] = image.bitmap.data[i + 2];  // B value
        inputImageData.data[i + 3] = image.bitmap.data[i + 3];  // A value
    }
    (uploadBtn! as HTMLButtonElement).style.display = "none";
    contextInput.putImageData(inputImageData, 0, 0);
});

const startBtn = document.querySelector("#start");
startBtn!.addEventListener("click", async () => {
    // Show spin loading
    const loading = document.querySelector("#loading");
    (loading! as HTMLDivElement).style.display = "flex";

    // Load model and do inference
    const model = await loadModel("./model/mosaic.onnx");
    const result = model.foward(tensor)[0];

    // [3, 224, 224] -> [224, 224, 3]
    let redChannel = [];
    let greenChannel = [];
    let blueChannel = [];
    for (let i = 0; i < result.data.length; i++) {
        if (i >= 2 * CHANNEL) {
            blueChannel.push(result.data[i]);
        } else if (i >= CHANNEL) {
            greenChannel.push(result.data[i]);
        } else {
            redChannel.push(result.data[i]);
        }
    }

    // Display image
    const canvasOutput = document.querySelector("#output-img")! as HTMLCanvasElement;
    const contextOutput = canvasOutput.getContext("2d")!;
    let outputImageData = contextOutput.createImageData(WIDTH, HEIGHT);
    for (let i = 0; i < CHANNEL; i++) {
        outputImageData.data[4 * i + 0] = redChannel[i];
        outputImageData.data[4 * i + 1] = greenChannel[i];
        outputImageData.data[4 * i + 2] = blueChannel[i];
        outputImageData.data[4 * i + 3] = 255;
    }
    (loading! as HTMLDivElement).style.display = "none";
    contextOutput.putImageData(outputImageData, 0, 0);
});
