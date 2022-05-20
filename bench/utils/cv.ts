import * as Jimp from "jimp";

export type Image = Jimp;

export interface Channels {
  red: number[];
  green: number[];
  blue: number[];
}

export async function loadImage(buffer: Buffer): Promise<Image> {
  return Jimp.default.read(buffer);
}

export function normalize(arr: Float32Array, factor: number) {
  return arr.map((x) => x / factor);
}

export function imageToArray(image: Jimp): Float32Array {
  const imageBufferData = image.bitmap.data;
  const dims = [1, 3, image.getWidth(), image.getHeight()];
  const [redChannel, greenChannel, blueAChannel] = new Array(
    new Array<number>(),
    new Array<number>(),
    new Array<number>()
  );
  for (let i = 0; i < imageBufferData.length; i += 4) {
    // skip the alpha channel
    redChannel.push(imageBufferData[i]);
    greenChannel.push(imageBufferData[i + 1]);
    blueAChannel.push(imageBufferData[i + 2]);
  }

  // [224, 224, 3] -> [3, 224, 224]
  const transposedData = redChannel.concat(greenChannel).concat(blueAChannel);

  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (let index = 0; index < transposedData.length; index++) {
    float32Data[index] = transposedData[index];
  }

  return float32Data;
}

export function displayImage(canvas: HTMLCanvasElement, image: Jimp) {
  const context = canvas.getContext("2d")!;
  const imageData = context.createImageData(
    image.getWidth(),
    image.getHeight()
  );
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i + 0] = image.bitmap.data[i + 0]; // R value
    imageData.data[i + 1] = image.bitmap.data[i + 1]; // G value
    imageData.data[i + 2] = image.bitmap.data[i + 2]; // B value
    imageData.data[i + 3] = image.bitmap.data[i + 3]; // A value
  }
  context.putImageData(imageData, 0, 0);
}

export function displayImageWithChannels(
  canvas: HTMLCanvasElement,
  channels: Channels,
  width: number,
  height: number
) {
  const contextOutput = canvas.getContext("2d")!;
  let outputImageData = contextOutput.createImageData(width, height);

  const length = width * height;
  for (let i = 0; i < length; i++) {
    outputImageData.data[4 * i + 0] = channels.red[i];
    outputImageData.data[4 * i + 1] = channels.green[i];
    outputImageData.data[4 * i + 2] = channels.blue[i];
    outputImageData.data[4 * i + 3] = 255;
  }
  contextOutput.putImageData(outputImageData, 0, 0);
}
