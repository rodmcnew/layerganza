// import {gaussRandom} from '../math/random'
// import {assertIsNumber} from '../assert/assert'

export default class OutputLayer {
    constructor(nodeCount, activationFunction, optimizer) {
        this.nodeCount = nodeCount;
        this.outputs = new Float64Array(nodeCount);
        this.activationFunction = activationFunction.xToY;
        this.activationFunctionDerivative = activationFunction.yToSlope;
        this.errorGradients = new Float64Array(nodeCount);
        this.optimizer = optimizer;
    }

    setInputLayer(inputLayer) {
        this.inputLayer = inputLayer;
        this.inputCount = inputLayer.nodeCount;
        this.inputNodeCount = this.inputCount + 1;//Add 1 for the bias node
        this.weights = new Float64Array(this.nodeCount * this.inputNodeCount);
        for (var weightI = 0, weightLen = this.weights.length; weightI < weightLen; weightI++) {
            this.weights[weightI] = Math.random() - 0.5; //gaussRandom(); //@TODO do better?
        }
    }

    feedForward() {
        this.inputs = this.inputLayer.outputs;

        //Defining these locally speeds up the loop below by reducing object property access
        var inputNodeCount = this.inputNodeCount;
        var weights = this.weights;
        var inputs = this.inputs;
        var activationFunction = this.activationFunction;
        var outputs = this.outputs;
        var nodeCount = this.nodeCount;
        var inputCount = this.inputCount;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            var sum = 0;
            for (var inputI = 0; inputI < inputCount; inputI++) {
                // assertIsNumber(this.inputs[inputI], 'Input');
                sum += inputs[inputI] * weights[neuronI * inputNodeCount + inputI];
            }
            sum += weights[neuronI * inputNodeCount + inputCount];//Bias node that always inputs "1"

            outputs[neuronI] = activationFunction(sum);

            // assertIsNumber(this.outputs[neuronI], 'Neuron output');
        }

        return outputs;
    }

    backPropagateCalculateErrorGradient(targetOutputs) {
        //Defining these locally speeds up the loop below by reducing object property access
        var nodeCount = this.nodeCount;
        var errorGradients = this.errorGradients;
        var outputs = this.outputs;
        var activationFunctionDerivative = this.activationFunctionDerivative;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            errorGradients[neuronI] = (outputs[neuronI] - targetOutputs[neuronI])
                * activationFunctionDerivative(outputs[neuronI]);
            // assertIsNumber(this.errorGradients[neuronI], 'Error gradient');
        }
    }

    backPropagateOptimize() {
        //Defining these locally speeds up the loop below by reducing object property access
        var inputNodeCount = this.inputNodeCount;
        var weights = this.weights;
        var nodeCount = this.nodeCount;
        var inputCount = this.inputCount;
        var inputs = this.inputs;
        var errorGradients = this.errorGradients;
        var calculateWeightUpdate = this.optimizer.calculateUpdate;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            for (var inputI = 0; inputI < inputCount; inputI++) {
                weights[neuronI * inputNodeCount + inputI] +=
                    calculateWeightUpdate(
                        inputs[inputI] * errorGradients[neuronI],
                        neuronI * inputNodeCount + inputI
                    );

                // assertIsNumber(this.weights[neuronI * this.inputNodeCount + inputI], 'Weight');
            }
            weights[neuronI * inputNodeCount + inputCount] += //Do the bias node weight
                calculateWeightUpdate(
                    errorGradients[neuronI],
                    neuronI * inputNodeCount + inputCount
                );

            // assertIsNumber(this.weights[neuronI * this.inputNodeCount + this.inputCount], 'Bias weight');
        }
    }
}
