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
        this.weightErrorGradients = new Float64Array(this.nodeCount * this.inputNodeCount);
        for (var weightI = 0, weightLen = this.weights.length; weightI < weightLen; weightI++) {
            this.weights[weightI] = Math.random() - 0.5; //@TODO would a gaussian distribution work better?
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
                sum += inputs[inputI] * weights[neuronI * inputNodeCount + inputI];
            }
            sum += weights[neuronI * inputNodeCount + inputCount];//Bias node that always inputs "1"

            outputs[neuronI] = activationFunction(sum);
        }

        return outputs;
    }

    backPropagateCalculateErrorGradient(targetOutputs) {
        this.calculateActivationErrorGradients(targetOutputs);
        this.calculateWeightErrorGradients();
    }

    calculateActivationErrorGradients(targetOutputs) {
        //Defining these locally speeds up the loop below by reducing object property access
        var nodeCount = this.nodeCount;
        var errorGradients = this.errorGradients;
        var outputs = this.outputs;
        var activationFunctionDerivative = this.activationFunctionDerivative;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            errorGradients[neuronI] = (outputs[neuronI] - targetOutputs[neuronI])
                * activationFunctionDerivative(outputs[neuronI]);
        }
    }

    calculateWeightErrorGradients() {
        //Defining these locally speeds up the loop below by reducing object property access
        var inputNodeCount = this.inputNodeCount;
        var nodeCount = this.nodeCount;
        var inputCount = this.inputCount;
        var inputs = this.inputs;
        var errorGradients = this.errorGradients;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            for (var inputI = 0; inputI < inputCount; inputI++) {
                this.weightErrorGradients[neuronI * inputNodeCount + inputI] = inputs[inputI] * errorGradients[neuronI];
            }
            this.weightErrorGradients[neuronI * inputNodeCount + inputI] = errorGradients[neuronI];
        }
    }

    backPropagateOptimize() {
        this.optimizer.optimizeWeights(this.weights, this.weightErrorGradients);
    }
}
