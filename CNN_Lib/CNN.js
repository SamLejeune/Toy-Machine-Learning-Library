import Matrix from "./matrix.js";
import DenseLayer from "./layers/denseLayer.js";
import MaxpoolLayer from "./layers/maxpoolLayer.js";
import ConvolutionalLayer from "./layers/convolutionalLayer.js";

export default class CNN {
  denseLayers = [];
  learningRate = 0.1;
  convolutionalLayers = [];
  denseLayers = [];

  constructor(optionsArr) {
    for (let i = 0; i < optionsArr.length; i++) {
      const optionObj = optionsArr[i];

      this.setConvolutionalLayers(optionObj);
    }
  }

  train(inputArr, targetArr) {
    this.feedForward(inputArr);

    const targetMatrix = Matrix.fromArrayToVector(targetArr);
    const output = this.denseLayers[this.denseLayers.length - 1].output;

    let outputErrorMatrix = Matrix.subtract(targetMatrix, output);

    for (let i = this.denseLayers.length - 1; i >= 0; i--) {
      const layer = this.denseLayers[i];
      outputErrorMatrix = layer.train(
        outputErrorMatrix,
        this.sigmoidDeriv,
        this.learningRate
      );
    }

    for (let i = this.convolutionalLayers.length - 1; i >= 0; i--) {
      const layer = this.convolutionalLayers[i];
      outputErrorMatrix = layer.train(outputErrorMatrix);
    }
  }

  feedForward(imageArr) {
    let input = imageArr;

    for (let i = 0; i < this.convolutionalLayers.length; i++) {
      const layer = this.convolutionalLayers[i];
      input = layer.feedForward(input);
    }

    let flatInput = [];

    for (let i = 0; i < input.length; i++) {
      const flat = Matrix.toArray(input[i]);
      flatInput = [...flatInput, ...flat];
    }

    input = flatInput;

    if (!(this.denseLayers[0] instanceof DenseLayer)) this.setDenseLayer(input);

    input = Matrix.fromArrayToVector(input);

    for (let i = 0; i < this.denseLayers.length; i++) {
      const layer = this.denseLayers[i];
      input = layer.feedForward(input, this.sigmoidActivation);
    }

    return input.data;
  }

  sigmoidActivation(x) {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDeriv(x) {
    return x * (1 - x);
  }

  softMaxActivation(outputVector) {
    const outputVectorMapBaseLog = Matrix.map(outputVector, this.calcBaseLog);
    const sumOutputVector = Matrix.sumMatrixValues(outputVectorMapBaseLog);

    outputVectorMapBaseLog.divide(sumOutputVector);

    const outputArr = Matrix.toArray(outputVectorMapBaseLog);

    return outputArr;
  }

  calcBaseLog(x) {
    return Math.exp(x);
  }

  setConvolutionalLayers(optionsObj) {
    const { type } = optionsObj;

    if (type === "convolutional") {
      const { number, rows, columns, stride } = optionsObj.kernel;

      const convLayer = new ConvolutionalLayer(number, rows, columns, stride);

      this.convolutionalLayers.push(convLayer);
    } else if (type === "maxpool") {
      const { rows, columns, stride } = optionsObj;

      const maxpoolLayer = new MaxpoolLayer(rows, columns, stride);

      this.convolutionalLayers.push(maxpoolLayer);
    } else {
      this.denseLayers.push(optionsObj);
    }
  }

  setDenseLayer(inputArr) {
    let prevLayerNeurons = inputArr.length;

    for (let i = 0; i < this.denseLayers.length; i++) {
      const optionObj = this.denseLayers[i];

      this.denseLayers[i] = new DenseLayer(optionObj.neurons, prevLayerNeurons);
      prevLayerNeurons = optionObj.neurons;
    }
  }
}
