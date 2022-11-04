import Matrix from "./matrix.js";

export default class Layer {
  type;
  weights;
  stride;
  bias;
  input;
  output;

  constructor(type, numRows, numColumns, stride) {
    this.type = type;
    this.weights = new Matrix(numRows, numColumns);
    this.stride = stride;
    this.bias = new Matrix(numRows, 1);
  }

  setInput(input) {
    this.input = input;
  }

  setChunkInput(chunkArr) {
    this.chunkInput = chunkArr;
  }

  setOutput(output) {
    this.output = output;
  }
}
