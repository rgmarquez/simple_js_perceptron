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

// Derivative of the sigmoid function
// 0.25     _
//        /   \
// 0.0   /     \
//       0     1
function sigmoidDerivative(x) {
  return x * (1 - x);
}

// Initialize weights and biases
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

function train(inputs, expectedOutputs, learningRate) {
  // Forward pass
  let hiddenLayerInput = [];
  for (let i = 0; i < 2; i++) {
    hiddenLayerInput[i] = 0;
    for (let j = 0; j < 2; j++) {
      hiddenLayerInput[i] += inputs[j] * weightsInputHidden[j][i];
    }
    hiddenLayerInput[i] += biasesHidden[i];
    hiddenLayerInput[i] = sigmoid(hiddenLayerInput[i]);
  }

  let outputLayerInput = [];
  for (let i = 0; i < 2; i++) {
    outputLayerInput[i] = 0;
    for (let j = 0; j < 2; j++) {
      outputLayerInput[i] += hiddenLayerInput[j] * weightsHiddenOutput[j][i];
    }
    outputLayerInput[i] += biasesOutput[i];
    outputLayerInput[i] = sigmoid(outputLayerInput[i]);
  }

  // Backward pass (calculate errors and update weights)
  let outputErrors = [];
  for (let i = 0; i < 2; i++) {
    outputErrors[i] = expectedOutputs[i] - outputLayerInput[i];
  }

  let hiddenErrors = [];
  for (let i = 0; i < 2; i++) {
    hiddenErrors[i] = 0;
    for (let j = 0; j < 2; j++) {
      hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j];
    }
    hiddenErrors[i] *= sigmoidDerivative(hiddenLayerInput[i]);
  }

  // Update weights and biases
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      weightsHiddenOutput[j][i] +=
        learningRate * outputErrors[i] * hiddenLayerInput[j];
    }
    biasesOutput[i] += learningRate * outputErrors[i];
  }

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      weightsInputHidden[j][i] += learningRate * hiddenErrors[i] * inputs[j];
    }
    biasesHidden[i] += learningRate * hiddenErrors[i];
  }
}

function predict(inputs) {
  let hiddenLayerInput = [];
  for (let i = 0; i < 2; i++) {
    hiddenLayerInput[i] = 0;
    for (let j = 0; j < 2; j++) {
      hiddenLayerInput[i] += inputs[j] * weightsInputHidden[j][i];
    }
    hiddenLayerInput[i] += biasesHidden[i];
    hiddenLayerInput[i] = sigmoid(hiddenLayerInput[i]);
  }

  let outputLayerInput = [];
  for (let i = 0; i < 2; i++) {
    outputLayerInput[i] = 0;
    for (let j = 0; j < 2; j++) {
      outputLayerInput[i] += hiddenLayerInput[j] * weightsHiddenOutput[j][i];
    }
    outputLayerInput[i] += biasesOutput[i];
    outputLayerInput[i] = sigmoid(outputLayerInput[i]);
  }

  return outputLayerInput;
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
const epochs = 10000;
for (let epoch = 0; epoch < epochs; epoch++) {
  for (let data of trainingData) {
    train(data.inputs, data.outputs, learningRate);
  }
}

// Predict based on command line inputs
const outputs = predict([input1, input2]);
console.log(`AND: ${outputs[0] > 0.5 ? 1 : 0}`);
console.log(`OR: ${outputs[1] > 0.5 ? 1 : 0}`);
