import Matrix from "../matrix.js";

export default class MaxpoolLayer {
  rows;
  columns;
  stride;
  input;
  output;
  chunkInput;

  constructor(numRows, numColumns, stride) {
    this.rows = numRows;
    this.columns = numColumns;
    this.stride = stride;
  }

  feedForward(input) {
    this.setInput(input);
    this.setOutput(input.length);
    this.setChunkInput(input.length);

    const inRows = input[0].rows;
    const inCols = input[0].columns;

    const mPoolRows = (inRows - this.rows) / this.stride + 1;
    const mPoolCols = (inCols - this.columns) / this.stride + 1;

    for (let i = 0; i < input.length; i++) {
      const inArr = input[i].data;

      const chunkInArr = this.chunkArr(inArr);
      const { maxArr, modArr } = Matrix.findMaxElseZero(chunkInArr);

      const maxPool = Matrix.fromArrayToMatrix(maxArr, mPoolRows, mPoolCols);

      this.output[i] = maxPool;
      this.chunkInput[i] = modArr;
    }

    return this.output;
  }

  train(outputError) {
    const inRows = this.input[0].rows;
    const inCols = this.input[0].columns;

    const outRows = this.output[0].rows;
    const outCols = this.output[0].columns;
    const outputLength = outRows * outCols;

    let inputError = new Array(this.input.length);
    let outErr =
      outputError instanceof Array
        ? this.flattenToVector(outputError)
        : outputError;

    for (let h = 0; h < this.chunkInput.length; h++) {
      const chunkArr = this.chunkInput[h];

      const sliceStart = h * outputLength;
      const sliceEnd = sliceStart + outputLength;
      const errorSlice = outErr.data.slice(sliceStart, sliceEnd);

      const errorArr = Matrix.toArray(errorSlice);
      const outputArr = Matrix.toArray(this.output[h]);

      for (let i = 0; i < chunkArr.length; i++) {
        const chunk = chunkArr[i];

        outer: for (let j = 0; j < chunk.rows; j++) {
          for (let k = 0; k < chunk.columns; k++) {
            if (chunk.data[k][j] !== outputArr[i]) continue;
            chunk.data[k][j] = errorArr[i];

            break outer;
          }
        }
      }

      const reconst = Matrix.reconstructMatrix(
        chunkArr,
        this.stride,
        inRows,
        inCols
      );

      inputError[h] = reconst;
    }

    return inputError;
  }

  chunkArr(arr, stride = this.stride, rows = this.rows, cols = this.columns) {
    return Matrix.deconstructMatrix(arr, stride, rows, cols);
  }

  flattenToVector(arr) {
    let flatArr = [];

    for (let i = 0; i < arr.length; i++) {
      const flat = Matrix.toArray(arr[i]);
      flatArr.push(...flat);
    }

    return Matrix.fromArrayToVector(flatArr);
  }

  setInput(input) {
    this.input = input;
  }

  setChunkInput(length) {
    this.chunkInput = new Array(length);
  }

  setOutput(length) {
    this.output = new Array(length);
  }
}
