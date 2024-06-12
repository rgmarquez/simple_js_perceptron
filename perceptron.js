// perceptron.js
// - Compute logical AND and OR of two inputs, values : 0.0, 1.0
// - EXAMPLE : node perceptron.js 0.0 1.0
//
// - Two input nodes
// - One hidden layer with two hidden nodes
// - Two output nodes
// - Data model:
//     input  weightsInputHidden     hidden     weightsHiddenOutput     output
//     [A: ]     {w1, w2} --       [B: , A: ]      {w1, w2} --        [B: , A: ]
//                         \                               \
//                         /                               /
//     [A: ]     {w1, w2} --       [B: , A: ]      {w1, w2} --        [B: , A: ]

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

// Initialize the Global NN model:
// Weights, Biases, Activations
let weightsInputHidden = [
  [Math.random(), Math.random()], // Weights from input to hidden neuron 1
  [Math.random(), Math.random()], // Weights from input to hidden neuron 2
];
let weightsHiddenOutput = [
  [Math.random(), Math.random()], // Weights from hidden neuron 1 to output neuron 1
  [Math.random(), Math.random()], // Weights from hidden neuron 2 to output neuron 2
];
let biasesHidden = [Math.random(), Math.random()];
let biasesOutput = [Math.random(), Math.random()];

let hiddenLayerActivation = [];
let outputLayerActivation = [];

function predict(inputs) {
  for (let i = 0; i < 2; i++) {
    hiddenLayerActivation[i] = 0;
    for (let j = 0; j < 2; j++) {
      hiddenLayerActivation[i] += inputs[j] * weightsInputHidden[j][i];
    }
    hiddenLayerActivation[i] += biasesHidden[i];
    hiddenLayerActivation[i] = sigmoid(hiddenLayerActivation[i]);
  }

  for (let i = 0; i < 2; i++) {
    outputLayerActivation[i] = 0;
    for (let j = 0; j < 2; j++) {
      outputLayerActivation[i] +=
        hiddenLayerActivation[j] * weightsHiddenOutput[j][i];
    }
    outputLayerActivation[i] += biasesOutput[i];
    outputLayerActivation[i] = sigmoid(outputLayerActivation[i]);
  }

  return outputLayerActivation;
}

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

function train(inputs, expectedOutputs, learningRate) {
  // Forward pass
  predict(inputs);

  // Calculate output layer errors with activation function derivative
  let outputErrors = [];
  for (let i = 0; i < 2; i++) {
    let error = expectedOutputs[i] - outputLayerActivation[i];
    outputErrors[i] = error * sigmoidDerivative(outputLayerActivation[i]);
  }

  // Calculate hidden layer errors
  let hiddenErrors = [];
  for (let i = 0; i < 2; i++) {
    hiddenErrors[i] = 0;
    for (let j = 0; j < 2; j++) {
      hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j];
    }
    hiddenErrors[i] *= sigmoidDerivative(hiddenLayerActivation[i]);
  }

  // Update weights and biases for hidden->output connections
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      weightsHiddenOutput[j][i] +=
        learningRate * outputErrors[i] * hiddenLayerActivation[j];
    }
    biasesOutput[i] += learningRate * outputErrors[i];
  }

  // Update weights and biases for input->hidden connections
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      weightsInputHidden[j][i] += learningRate * hiddenErrors[i] * inputs[j];
    }
    biasesHidden[i] += learningRate * hiddenErrors[i];
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

// Train the model
const learningRate = 0.1;
//const epochs = 10000;
const epochs = 4000;
for (let epoch = 0; epoch < epochs; epoch++) {
  for (let data of trainingData) {
    train(data.inputs, data.outputs, learningRate);
  }
}

// Predict based on command line inputs
const outputs = predict([input1, input2]);
console.log(`AND: ${outputs[0] > 0.5 ? 1 : 0}`);
console.log(`OR: ${outputs[1] > 0.5 ? 1 : 0}`);
