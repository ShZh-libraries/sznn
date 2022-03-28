import * as Jimp from "jimp";
import { DType, Tensor, TensorBuilder } from "../js/tensor";

export type Image = Jimp;

export async function fetchImage(path: string): Promise<Image> {
  // var imageData = await Jimp.default.read(path).then((imageBuffer: Image) => {
  //   return imageBuffer.resize(width, height);
  // });  
  return Jimp.default.read(path);
}

export async function loadImage(buffer: Buffer): Promise<Image> {
  return Jimp.default.read(buffer);
}

export function imageToTensor(image: Jimp): Tensor {
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

  const resultTensor = TensorBuilder.withAllArgs(float32Data, dims, DType.float32);
  
  return resultTensor;
}