import Matrix from "../matrix.js";

export default class DenseLayer {
  weights;
  bias;
  input;
  output;

  constructor(numRows, numColumns) {
    this.weights = new Matrix(numRows, numColumns);
    this.bias = new Matrix(numRows, 1);
  }

  feedForward(input, activationFunc) {
    const outputMatrix = Matrix.multiply(this.weights, input);
    outputMatrix.add(this.bias);
    outputMatrix.map(activationFunc);

    this.setInput(input);
    this.setOutput(outputMatrix);

    return outputMatrix;
  }

  train(outputError, activationFuncDeriv, learningRate) {
    let gradientMatrix = Matrix.map(this.output, activationFuncDeriv);

    gradientMatrix.multiply(outputError);
    gradientMatrix.multiply(learningRate);

    const transInput = Matrix.transpose(this.input);
    const deltaWeights = Matrix.multiply(gradientMatrix, transInput);

    this.weights.add(deltaWeights);
    this.bias.add(gradientMatrix);

    const currWeightsTrans = Matrix.transpose(this.weights);
    const outputErrorMatrix = Matrix.multiply(currWeightsTrans, outputError);

    return outputErrorMatrix;
  }

  setInput(input) {
    this.input = input;
  }

  setOutput(output) {
    this.output = output;
  }
}
