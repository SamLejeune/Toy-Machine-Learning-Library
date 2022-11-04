import Matrix from "../matrix.js";

export default class ConvolutionalLayer {
  kernelArr;
  biasArr;
  kernelRows;
  kernelCols;
  stride;
  input;
  output;

  constructor(num, numRows, numColumns, stride) {
    this.kernelRows = numRows;
    this.kernelCols = numColumns;
    this.stride = stride;

    this.setKernelArr(num);
    this.setBiasArr(num);
  }

  feedForward(inputArr) {
    this.setInput(inputArr);
    this.setOutput();

    for (let h = 0; h < inputArr.length; h++) {
      const input =
        inputArr[h] instanceof Matrix ? inputArr[h].data : inputArr[h];

      const inRows = input instanceof Matrix ? input.rows : input.length;
      const inCols = input instanceof Matrix ? input.columns : input[0].length;
      const convRows = this.calcOutputRows(inRows);
      const convCols = this.calcOutputCols(inCols);

      const chunkInArr = this.chunkArr(input);

      let convolution;

      for (let i = 0; i < this.kernelArr.length; i++) {
        if (!this.biasArr[i]) this.initBias(convRows, convCols, i);

        convolution = new Matrix(convRows, convCols);
        const kernel = this.kernelArr[i];

        for (let j = 0; j < convRows; j++) {
          for (let k = 0; k < convCols; k++) {
            const chunkArr = chunkInArr[j * convRows + k];
            convolution.data[j][k] = Matrix.convolve(chunkArr, kernel);
          }
        }

        this.output[h * inputArr.length + i] = convolution;
      }
    }
    return this.output;
  }

  train(outputError) {
    let inputError;
    let inIndex = 0;
    let kerIndex = 0;

    const numInput = this.input.length;

    if (this.input[0] instanceof Matrix)
      inputError = new Array(this.input.length);

    for (let h = 0; h < outputError.length; h++) {
      if (h > 0 && numInput > 1 && h % numInput === 0) inIndex++;
      if (h > 0 && numInput > 1 && h % numInput === 0) kerIndex++;

      const error = outputError[h];
      const kernel =
        numInput > 1 ? this.kernelArr[kerIndex] : this.kernelArr[h];
      const input =
        this.input[inIndex] instanceof Matrix
          ? this.input[inIndex].data
          : this.input[inIndex];

      const chunkInputArr = this.chunkArr(
        input,
        this.stride,
        error.rows,
        error.columns
      );

      if (inputError) {
        if (h % numInput === 0)
          inputError[inIndex] = this.calcInputError(error, kernel);
        if (h % numInput !== 0) {
          inputError[inIndex].add(this.calcInputError(error, kernel));
        }
      }

      const deltaWeights = new Matrix(this.kernelRows, this.kernelCols);

      for (let i = 0; i < deltaWeights.rows; i++) {
        for (let j = 0; j < deltaWeights.columns; j++) {
          const chunkArr = chunkInputArr[i * deltaWeights.rows + j];
          const delta = Matrix.convolve(chunkArr, error);

          deltaWeights.data[i][j] = delta;
        }
      }

      kernel.add(deltaWeights);

      this.biasArr[h] = error;
      // kernel.multiply(0.8);
    }

    return inputError;
  }

  calcInputError(outputError, kernel) {
    const padRows = this.calcPadRows(outputError.rows);
    const padCols = this.calcPadRows(outputError.columns);

    const padOuputError = Matrix.pad(outputError, padRows, padCols);

    const chunkOutputError = this.chunkArr(padOuputError.data);

    const inRows = this.input[0].rows;
    const inCols = this.input[0].columns;
    let inputError = new Matrix(inRows, inCols);

    for (let i = 0; i < inRows; i++) {
      for (let j = 0; j < inCols; j++) {
        const chunkArr = chunkOutputError[i * inRows + j];

        inputError.data[i][j] = Matrix.convolve(chunkArr, kernel);
      }
    }

    return inputError;
  }

  chunkArr(
    arr,
    stride = this.stride,
    rows = this.kernelRows,
    cols = this.kernelCols
  ) {
    return Matrix.deconstructMatrix(arr, stride, rows, cols);
  }

  calcPadRows(outputRows) {
    return (
      (this.input[0].rows - 1) * this.stride + this.kernelRows - outputRows
    );
  }

  calcOutputRows(inputRows) {
    return (inputRows - this.kernelRows) / this.stride + 1;
  }

  calcOutputCols(inputCols) {
    return (inputCols - this.kernelCols) / this.stride + 1;
  }

  setKernelArr(num) {
    this.kernelArr = new Array(num);

    for (let i = 0; i < num; i++) {
      this.kernelArr[i] = new Matrix(this.kernelRows, this.kernelCols);
    }
  }

  initBias(outputRows, outputCols, i) {
    this.biasArr[i] = new Matrix(outputRows, outputCols);
  }

  setBiasArr(num) {
    this.biasArr = new Array(num);
  }

  setInput(input) {
    this.input = input;
  }

  setOutput() {
    this.output = new Array(this.input.length * this.kernelArr.length);
  }
}
