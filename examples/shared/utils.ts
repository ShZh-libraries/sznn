import Jimp from "jimp/*";

export interface Channels {
  red: number[];
  green: number[];
  blue: number[];
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
