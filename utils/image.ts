import * as Jimp from "jimp";
import { Tensor, TensorBuilder } from "../js/tensor";

export type Image = Jimp;

export async function fetchImage(path: string, width: number=224, height: number=224): Promise<Image> {
  var imageData = await Jimp.default.read(path).then((imageBuffer: Image) => {
    return imageBuffer.resize(width, height);
  });

  return imageData;
}

export function imageToTensor(image: Jimp, dims: number[]): Tensor {
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = new Array(
    new Array<number>(), 
    new Array<number>(), 
    new Array<number>()
  );
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // skip the alpha channel
  }

  // [224, 224, 3] -> [3, 224, 224]
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (let index = 0; index < transposedData.length; index++) {
    float32Data[index] = transposedData[index] / 255.0; // convert to float
  }

  const resultTensor = TensorBuilder.withAllArgs(float32Data, dims);
  
  return resultTensor;
}