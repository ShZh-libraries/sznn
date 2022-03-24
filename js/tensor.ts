type TensorDataType = 
    Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array | Uint8ClampedArray |
    Float32Array | Float64Array;

enum DType {
    int8,
    int16,
    int32,
    uint8,
    uint16,
    uint32,
    float32,
    float64,
}

type InputArray = number[] | number[][] | number[][][] | number[][][][];

export class Tensor {
    data!: TensorDataType;
    shape: number[];
    ndim: number;
    dtype: DType;

    constructor(data: InputArray, shape: number | number[], dtype?: DType) {
        if (!Array.isArray(shape)) {
            shape = Array.of(shape);
        }
        this.shape = shape;
        this.ndim = shape.length;
        this.dtype = dtype || DType.float32;

        this.flattenInput(data);
    }

    private flattenInput(data: InputArray) {
        const dataLength = this.shape.reduceRight((x, y) => x * y);
        switch(this.dtype) {
            case DType.int8: this.data = new Int8Array(dataLength); break;
            case DType.int16: this.data = new Int16Array(dataLength); break;
            case DType.int32: this.data = new Int32Array(dataLength); break;
            case DType.uint8: this.data = new Uint8Array(dataLength); break;
            case DType.uint16: this.data = new Uint16Array(dataLength); break;
            case DType.uint32: this.data = new Uint32Array(dataLength); break;
            case DType.float32: this.data = new Float32Array(dataLength); break;
            case DType.float64: this.data = new Float64Array(dataLength); break;
        }

        switch(this.ndim) {
            case 1: flattenTensor1d(this.data, data as number[], this.shape); break;
            case 2: flattenTensor2d(this.data, data as number[][], this.shape); break;
            case 3: flattenTensor3d(this.data, data as number[][][], this.shape); break;
            case 4: flattenTensor4d(this.data, data as number[][][][], this.shape); break;
        }
    }
}

function flattenTensor1d(dst: TensorDataType, src: number[], shape: number[]) {
    for (let i = 0; i < shape[0]; i++) {
        dst[i] = src[i];
    }
}

function flattenTensor2d(dst: TensorDataType, src: number[][], shape: number[]) {
    let index = 0;
    for (let i = 0; i < shape[0]; i++) {
        for (let j = 0; j < shape[1]; j++) {
            dst[index++] = src[i][j];
        }
    }
}

function flattenTensor3d(dst: TensorDataType, src: number[][][], shape: number[]) {
    let index = 0;
    for (let i = 0; i < shape[0]; i++) {
        for (let j = 0; j < shape[1]; j++) {
            for (let k = 0; k < shape[2]; k++) {
                dst[index++] = src[i][j][k];
            }
        }
    }
}

function flattenTensor4d(dst: TensorDataType, src: number[][][][], shape: number[]) {
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