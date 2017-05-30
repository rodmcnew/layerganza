import OutputLayer from './OutputLayer'

export default class HiddenLayer extends OutputLayer {
    setOutputLayer(outputLayer) {
        this.outputLayer = outputLayer;
    }

    backPropagateCalculateErrorGradient() {
        //Defining these locally speeds up the loop below by reducing object property access
        var nodeCount = this.nodeCount;
        var errorGradients = this.errorGradients;
        var outputs = this.outputs;
        var activationFunctionDerivative = this.activationFunctionDerivative;
        var outputLayerNodeCount = this.outputLayer.nodeCount;
        var outputLayerWeights = this.outputLayer.weights;
        var outputLayerInputNodeCount = this.outputLayer.inputNodeCount;
        var outputLayerErrorGradients = this.outputLayer.errorGradients;
        var inputs = this.inputs;
        var inputCount = this.inputCount;
        var inputNodeCount = this.inputNodeCount;

        for (var neuronI = 0; neuronI < nodeCount; neuronI++) {
            var errorWithRespectToOutput = 0;
            for (var outputI = 0; outputI < outputLayerNodeCount; outputI++) {
                errorWithRespectToOutput += outputLayerErrorGradients[outputI]
                    * outputLayerWeights[outputI * outputLayerInputNodeCount + neuronI];

            }

            var activationErrorGradient = errorWithRespectToOutput
                * activationFunctionDerivative(outputs[neuronI]);
            errorGradients[neuronI] = activationErrorGradient;

            for (var inputI = 0; inputI < inputCount; inputI++) {
                this.weightErrorGradients[neuronI * inputNodeCount + inputI] = inputs[inputI] * activationErrorGradient;
            }
            this.weightErrorGradients[neuronI * inputNodeCount + inputI] = activationErrorGradient;//Bias node
        }
    }
}
