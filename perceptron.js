/**
 * A simple JS program to implement, train, and test a perceptron to perform
 * AND and OR calculations on two input numbers [0.0 .. 1.0]
 * @author Richard "Greg" Marquez (aka G-Money)
 * @license MIT
 * @example node perceptron.js 0.0 1.0
 * @description two input nodes, one hidden layer with two hidden nodes, two output nodes
 *     input  weightsInputHidden     hidden     weightsHiddenOutput     output
 *     [A: ]     {w1, w2} --       [B: , A: ]      {w1, w2} --        [B: , A: ]
 *                         \                               \
 *                         /                               /
 *     [A: ]     {w1, w2} --       [B: , A: ]      {w1, w2} --        [B: , A: ]
 */

const process = require("process");

// Activation function (Sigmoid)
// 1.0           --------
//             /
// 0.0  -------
//             0
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Derivative of the activation (sigmoid) function
// 0.25     _
//        /   \
// 0.0   /     \
//       0     1
function sigmoidDerivative(x) {
  return x * (1 - x);
}

/**
 * Class representing a simple perceptron with an input layer with two input nodes,
 * one hidden layer with two hidden nodes, and an output layer with two output nodes.
 */
class Perceptron {
  #weightsInputHidden;
  #weightsHiddenOutput;
  #biasesHidden;
  #biasesOutput;
  #hiddenLayerActivation;
  #outputLayerActivation;

  /**
   * Create a Perceptron.
   */
  constructor() {
    this.#weightsInputHidden = [
      [Math.random(), Math.random()], // Weights from input to hidden neuron 1
      [Math.random(), Math.random()], // Weights from input to hidden neuron 2
    ];
    this.#weightsHiddenOutput = [
      [Math.random(), Math.random()], // Weights from hidden neuron 1 to output neuron 1
      [Math.random(), Math.random()], // Weights from hidden neuron 2 to output neuron 2
    ];
    this.#biasesHidden = [Math.random(), Math.random()];
    this.#biasesOutput = [Math.random(), Math.random()];

    this.#hiddenLayerActivation = [];
    this.#outputLayerActivation = [];
  }

  /**
   * Have the perceptron make a prediction based on its current state.
   * @param {number[]} inputs - The input values to run through the perceptron.
   * @return {number[]} The "answer" values from the perteptron.
   */
  predict(inputs) {
    for (let i = 0; i < 2; i++) {
      this.hiddenLayerActivation[i] = 0;
      for (let j = 0; j < 2; j++) {
        this.hiddenLayerActivation[i] +=
          inputs[j] * this.weightsInputHidden[j][i];
      }
      this.hiddenLayerActivation[i] += this.biasesHidden[i];
      this.hiddenLayerActivation[i] = sigmoid(this.hiddenLayerActivation[i]);
    }

    for (let i = 0; i < 2; i++) {
      this.outputLayerActivation[i] = 0;
      for (let j = 0; j < 2; j++) {
        this.outputLayerActivation[i] +=
          this.hiddenLayerActivation[j] * this.weightsHiddenOutput[j][i];
      }
      this.outputLayerActivation[i] += this.biasesOutput[i];
      this.outputLayerActivation[i] = sigmoid(this.outputLayerActivation[i]);
    }

    return this.outputLayerActivation;
  }

  /**
   * Train the perceptron using an example input, the perceptron's current state, the expected
   * output, and a learning rate to train the perceptron to get better at matching the expected
   * output.  Each call to train() is effectively calculating the next step in a min-batch
   * gradient descent, tweaking the weights and biases a little closer to matching the current
   * expected output.
   * @param {number[]} inputs - The input values to run through the perceptron.
   * @param {number[]} expectedOutputs - the expected outputs, used to train the perceptron.
   * @param {number} learningRate - a learning rate used to scale how much the perptron nudges
   *  the weights and biases durning learning, typically ~ 0.1
   */
  train(inputs, expectedOutputs, learningRate) {
    // Forward pass
    this.predict(inputs);

    // Gradient Descent is an optimization technique used to minimize the loss function
    // by iteratively adjusting the weights and biases in the direction of the steepest
    // descent (negative gradient).
    // We use the derivative (slope) of the activation function with respect to each activation
    // to find the direction of the steepest descent (negative gradient) in order to adjust
    // the weights to minimize the error
    // In mathematical terms, this is part of the chain rule used in backpropagation.
    // By multiplying the error by the derivative of the activation function, we are effectively
    // determining how much the weights leading into this hidden neuron should be adjusted to
    // reduce the overall error.

    // Calculate output layer errors using the activation function derivative/batch gradient descent
    let outputErrors = [];
    for (let i = 0; i < 2; i++) {
      let error = expectedOutputs[i] - this.outputLayerActivation[i];
      outputErrors[i] =
        error * sigmoidDerivative(this.outputLayerActivation[i]);
    }

    // Calculate hidden layer errors using the activation function derivative/batch gradient descent
    let hiddenErrors = [];
    for (let i = 0; i < 2; i++) {
      hiddenErrors[i] = 0;
      for (let j = 0; j < 2; j++) {
        hiddenErrors[i] += outputErrors[j] * this.weightsHiddenOutput[i][j];
      }
      hiddenErrors[i] *= sigmoidDerivative(this.hiddenLayerActivation[i]);
    }

    // Update weights and biases for hidden->output connections
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        this.weightsHiddenOutput[j][i] +=
          learningRate * outputErrors[i] * this.hiddenLayerActivation[j];
      }
      this.biasesOutput[i] += learningRate * outputErrors[i];
    }

    // Update weights and biases for input->hidden connections
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        this.weightsInputHidden[j][i] +=
          learningRate * hiddenErrors[i] * inputs[j];
      }
      this.biasesHidden[i] += learningRate * hiddenErrors[i];
    }
  }

  // JavaScript boilerplate for implenting the getters and setters
  get weightsInputHidden() {
    return this.#weightsInputHidden;
  }
  set weightsInputHidden(newValue) {
    this.#weightsInputHidden = newValue;
  }
  get weightsHiddenOutput() {
    return this.#weightsHiddenOutput;
  }
  set weightsHiddenOutput(newValue) {
    this.#weightsHiddenOutput = newValue;
  }
  get biasesHidden() {
    return this.#biasesHidden;
  }
  set biasesHidden(newValue) {
    this.#biasesHidden = newValue;
  }
  get biasesOutput() {
    return this.#biasesOutput;
  }
  set biasesOutput(newValue) {
    this.#biasesOutput = newValue;
  }
  get hiddenLayerActivation() {
    return this.#hiddenLayerActivation;
  }
  set hiddenLayerActivation(newValue) {
    this.#hiddenLayerActivation = newValue;
  }
  get outputLayerActivation() {
    return this.#outputLayerActivation;
  }
  set outputLayerActivation(newValue) {
    this.#outputLayerActivation = newValue;
  }
}

// Get inputs from command line
const input1 = parseFloat(process.argv[2]);
const input2 = parseFloat(process.argv[3]);

// Define the training data
const trainingData = [
  { inputs: [0, 0], outputs: [0, 0] },
  { inputs: [0, 1], outputs: [0, 1] },
  { inputs: [1, 0], outputs: [0, 1] },
  { inputs: [1, 1], outputs: [1, 1] },
];

// Create our perceptron
const perceptron = new Perceptron();

// Train the perceptron's model
const learningRate = 0.1;
const epochs = 4000;
for (let epoch = 0; epoch < epochs; epoch++) {
  for (let data of trainingData) {
    perceptron.train(data.inputs, data.outputs, learningRate);
  }
}

// Make a prediction of the correct answer for the command line inputs
const outputs = perceptron.predict([input1, input2]);
console.log(`AND: ${outputs[0] > 0.5 ? 1 : 0}`);
console.log(`OR: ${outputs[1] > 0.5 ? 1 : 0}`);
