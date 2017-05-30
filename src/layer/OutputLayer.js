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
        //Defining these locally speeds up the loop below by reducing object property access
        var nodeCount = this.nodeCount;
        var errorGradients = this.errorGradients;
        var outputs = this.outputs;
        var activationFunctionDerivative = this.activationFunctionDerivative;
        var inputNodeCount = this.inputNodeCount;
        var inputCount = this.inputCount;
        var inputs = this.inputs;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            var activationErrorGradient = (outputs[neuronI] - targetOutputs[neuronI])
                * activationFunctionDerivative(outputs[neuronI]);
            errorGradients[neuronI] = activationErrorGradient;

            for (var inputI = 0; inputI < inputCount; inputI++) {
                this.weightErrorGradients[neuronI * inputNodeCount + inputI] = inputs[inputI] * activationErrorGradient;
            }
            this.weightErrorGradients[neuronI * inputNodeCount + inputI] = activationErrorGradient;//Bias node
        }
    }

    backPropagateOptimize() {
        this.optimizer.optimizeWeights(this.weights, this.weightErrorGradients);
    }
}
