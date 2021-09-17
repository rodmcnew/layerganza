import OutputLayer from './OutputLayer'

export default class HiddenLayer extends OutputLayer {
    setOutputLayer(outputLayer: any) { //@TODO type
        this.outputLayer = outputLayer;
    }

    backPropagateCalculateErrorGradient() {
        //Defining these locally speeds up the loop below by reducing object property access
        let nodeCount = this.nodeCount;
        let errorGradients = this.errorGradients;
        let outputs = this.outputs;
        let activationFunctionDerivative = this.activationFunctionDerivative;
        let outputLayerNodeCount = this.outputLayer.nodeCount;
        let outputLayerWeights = this.outputLayer.weights;
        let outputLayerInputNodeCount = this.outputLayer.inputNodeCount;
        let outputLayerErrorGradients = this.outputLayer.errorGradients;
        let inputs = this.inputs;
        let inputCount = this.inputCount;
        let inputNodeCount = this.inputNodeCount;
        let weightErrorGradients = this.weightErrorGradients;

        // Defining these here ideally speeds up the loop below
        let inputI;
        let errorWithRespectToOutput;
        let activationErrorGradient;
        let outputI;

        for (let neuronI = 0; neuronI < nodeCount; neuronI++) {
            errorWithRespectToOutput = 0;
            for (outputI = 0; outputI < outputLayerNodeCount; outputI++) {
                errorWithRespectToOutput += outputLayerErrorGradients[outputI]
                    * outputLayerWeights[outputI * outputLayerInputNodeCount + neuronI];
            }

            activationErrorGradient = errorWithRespectToOutput
                * activationFunctionDerivative(outputs[neuronI]);
            errorGradients[neuronI] = activationErrorGradient;

            for (inputI = 0; inputI < inputCount; inputI++) {
                weightErrorGradients[neuronI * inputNodeCount + inputI] = inputs[inputI] * activationErrorGradient;
            }
            weightErrorGradients[neuronI * inputNodeCount + inputCount] = activationErrorGradient;//Bias node
        }
    }
}
