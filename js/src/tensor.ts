import { onnx } from "onnx-proto";

export enum DType {
  int8,
  int16,
  int32,
  uint8,
  uint16,
  uint32,
  float32,
  float64,
}

type TensorDataType =
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Float32Array
  | Float64Array;

type InputArray = number[] | number[][] | number[][][] | number[][][][];

export class Tensor {
  data!: TensorDataType;
  shape!: number[];
  ndim!: number;
  dtype!: DType;

  stride?: number[];

  allocateWithShape() {
    const dataLength = this.shape.reduceRight((x, y) => x * y);
    switch (this.dtype) {
      case DType.int8:
        this.data = new Int8Array(dataLength);
        break;
      case DType.int16:
        this.data = new Int16Array(dataLength);
        break;
      case DType.int32:
        this.data = new Int32Array(dataLength);
        break;
      case DType.uint8:
        this.data = new Uint8Array(dataLength);
        break;
      case DType.uint16:
        this.data = new Uint16Array(dataLength);
        break;
      case DType.uint32:
        this.data = new Uint32Array(dataLength);
        break;
      case DType.float32:
        this.data = new Float32Array(dataLength);
        break;
      case DType.float64:
        this.data = new Float64Array(dataLength);
        break;
    }
  }

  toArray() {
    return this.data;
  }

  locToIndex(location: number[]): number {
    if (!this.stride) {
      this.calcStride();
    }

    let index = 0;
    for (let i = 0; i < this.ndim; i++) {
      index += location[i] * this.stride![i];
    }

    return index;
  }

  indexToLoc(index: number): number[] {
    if (!this.stride) {
      this.calcStride();
    }

    let indexes = [];
    for (let i = 0; i < this.ndim; i++) {
      indexes.push(Math.floor(index / this.stride![i]));
      index %= this.stride![i];
    }

    return indexes;
  }

  atLoc(location: number[]): number {
    const index = this.locToIndex(location);
    return this.data[index];
  }

  reshape(shape: number[]) {
    this.shape = shape.slice();
    this.ndim = this.shape.length;

    return this;
  }

  normalize(factor: number) {
    for (let index = 0; index < this.data.length; index++) {
      this.data[index] /= factor;
    }

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

  calcStride() {
    this.stride = [1];
    for (let i = 1; i < this.ndim; i++) {
      this.stride.unshift(this.stride[0] * this.shape[this.ndim - i]);
    }
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
        case 7: // Currently int64 type is not supported yet
          tensor.dtype = DType.int32;
          tensor.data = new Int32Array(buffer);
          break;
        case 11:
          tensor.dtype = DType.float64;
          tensor.data = new Float64Array(buffer);
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
          tensor.dtype = DType.float64;
          tensor.data = Float64Array.from(initializer.doubleData);
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
