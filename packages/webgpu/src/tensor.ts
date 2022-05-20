import { onnx } from "onnx-proto";

export enum DType {
  int32,
  uint32,
  float32,
}

type TensorDataType = Int32Array | Uint32Array | Float32Array;

type InputArray = number[] | number[][] | number[][][] | number[][][][];

export class Tensor {
  data!: TensorDataType;
  shape!: number[];
  ndim!: number;
  dtype!: DType;

  stride?: number[];

  toArray() {
    return this.data;
  }

  reshape(shape: number[]) {
    this.shape = shape.slice();
    this.ndim = this.shape.length;

    return this;
  }

  copy() {
    let tensor = new Tensor();
    tensor.shape = this.shape.slice();
    tensor.ndim = this.ndim;
    tensor.dtype = this.dtype;
    tensor.data = this.data.slice();

    return tensor;
  }

  getLength() {
    return this.shape.reduceRight((x, y) => x * y);
  }

  calcStride() {
    this.stride = [1];
    for (let i = 1; i < this.ndim; i++) {
      this.stride.unshift(this.stride[0] * this.shape[this.ndim - i]);
    }
  }

  getStride() {
    if (!this.stride) {
      this.calcStride();
    }

    return this.stride;
  }

  allocateWithShape() {
    const dataLength = this.getLength();
    switch (this.dtype) {
      case DType.int32:
        this.data = new Int32Array(dataLength);
        break;
      case DType.uint32:
        this.data = new Uint32Array(dataLength);
        break;
      case DType.float32:
        this.data = new Float32Array(dataLength);
        break;
    }
  }

  setInputGPUBuffer(device: GPUDevice): GPUBuffer {
    const gpuBuffer = device.createBuffer({
      mappedAtCreation: true,
      size: this.data.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    const arrayBuffer = gpuBuffer.getMappedRange();
    switch (this.dtype) {
      // case DType.int32:
      //     new Int32Array(arrayBuffer).set(this.data);
      //     break;
      // case DType.uint32:
      //     new Uint32Array(arrayBuffer).set(this.data);
      //     break;
      case DType.float32:
        new Float32Array(arrayBuffer).set(this.data);
        break;
      default:
        throw new Error("Type not support in Web GPU!");
    }

    gpuBuffer.unmap();

    return gpuBuffer;
  }

  setOutputGPUBuffer(device: GPUDevice): GPUBuffer {
    // All data type occupy 4 bytes
    const dataLen = this.getLength() * 4;

    const gpuBuffer = device.createBuffer({
      size: dataLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    return gpuBuffer;
  }
}

export class TensorBuilder {
  static withData(data: InputArray, dtype?: DType): Tensor {
    // Set normal attributes
    let tensor = new Tensor();
    tensor.shape = this.inferShape(data);
    tensor.ndim = tensor.shape.length;
    tensor.dtype = dtype || DType.float32;
    // Move input array to tensor.data field
    tensor.allocateWithShape();
    this.flattenInput(data, tensor);

    return tensor;
  }

  static withShape(shape: number[], dtype?: DType): Tensor {
    let tensor = new Tensor();
    tensor.shape = shape.slice();
    tensor.ndim = tensor.shape.length;
    tensor.dtype = dtype || DType.float32;
    // Initialize for data field
    tensor.allocateWithShape();

    return tensor;
  }

  static withAllArgs(
    data: TensorDataType,
    shape: number[],
    dtype?: DType
  ): Tensor {
    let tensor = new Tensor();
    tensor.data = data;
    tensor.shape = shape.slice();
    tensor.ndim = tensor.shape.length;
    tensor.dtype = dtype || DType.float32;

    return tensor;
  }

  static withInitializer(initializer: onnx.TensorProto): Tensor {
    let tensor = new Tensor();
    tensor.shape = initializer.dims as number[];
    tensor.ndim = tensor.shape.length;

    if (initializer.rawData.length != 0) {
      let buffer = initializer.rawData.buffer.slice(
        initializer.rawData.byteOffset,
        initializer.rawData.byteOffset + initializer.rawData.byteLength
      );
      switch (initializer.dataType) {
        case 1:
          tensor.dtype = DType.float32;
          tensor.data = new Float32Array(buffer);
          break;
        case 6:
        case 7:
          tensor.dtype = DType.int32;
          tensor.data = new Int32Array(buffer);
          break;
        case 11:
          tensor.dtype = DType.float32;
          tensor.data = new Float32Array(buffer);
          break;
        default:
          throw Error("Data type not support in ONNX!!");
      }
    } else {
      switch (initializer.dataType) {
        case 1:
          tensor.dtype = DType.float32;
          tensor.data = Float32Array.from(initializer.floatData);
          break;
        case 6:
          tensor.dtype = DType.int32;
          tensor.data = Int32Array.from(initializer.int32Data);
          break;
        case 7:
          tensor.dtype = DType.int32;
          tensor.data = Int32Array.from(initializer.int64Data as number[]);
          break;
        case 11:
          tensor.dtype = DType.float32;
          tensor.data = Float32Array.from(initializer.doubleData);
          break;
        default:
          throw Error("Data type not support in ONNX!!");
      }
    }

    return tensor;
  }

  private static inferShape(data: InputArray): number[] {
    let shape: number[] = [];
    let curObj = data;
    while (Array.isArray(curObj)) {
      shape.push(curObj.length);
      curObj = curObj[0] as InputArray;
    }

    return shape;
  }

  private static flattenInput(src: InputArray, dst: Tensor): void {
    if (src.length != 0) {
      switch (dst.ndim) {
        case 1:
          flattenTensor1d(dst.data, src as number[], dst.shape);
          break;
        case 2:
          flattenTensor2d(dst.data, src as number[][], dst.shape);
          break;
        case 3:
          flattenTensor3d(dst.data, src as number[][][], dst.shape);
          break;
        case 4:
          flattenTensor4d(dst.data, src as number[][][][], dst.shape);
          break;
      }
    }
  }
}

export class TensorDict {
  private pool: Map<string, Tensor> = new Map();

  get(name: string): Tensor | undefined {
    return this.pool.get(name);
  }

  set(name: string, tensor: Tensor): void {
    this.pool.set(name, tensor);
  }

  init(initializers: onnx.ITensorProto[]) {
    for (const initializer of initializers) {
      const tensor = TensorBuilder.withInitializer(
        initializer as onnx.TensorProto
      );
      this.pool.set((initializer as onnx.TensorProto).name, tensor);
    }

    return this;
  }
}

function flattenTensor1d(dst: TensorDataType, src: number[], shape: number[]) {
  for (let i = 0; i < shape[0]; i++) {
    dst[i] = src[i];
  }
}

function flattenTensor2d(
  dst: TensorDataType,
  src: number[][],
  shape: number[]
) {
  let index = 0;
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[1]; j++) {
      dst[index++] = src[i][j];
    }
  }
}

function flattenTensor3d(
  dst: TensorDataType,
  src: number[][][],
  shape: number[]
) {
  let index = 0;
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[1]; j++) {
      for (let k = 0; k < shape[2]; k++) {
        dst[index++] = src[i][j][k];
      }
    }
  }
}

function flattenTensor4d(
  dst: TensorDataType,
  src: number[][][][],
  shape: number[]
) {
  let index = 0;
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[1]; j++) {
      for (let k = 0; k < shape[2]; k++) {
        for (let l = 0; l < shape[3]; l++) {
          dst[index++] = src[i][j][k][l];
        }
      }
    }
  }
}
