export default class OutputLayer {
    protected inputs: Float64Array;
    public weights: Float64Array;
    protected weightErrorGradients: Float64Array;
    protected outputs: Float64Array;
    protected nodeCount: number;
    protected inputCount: number;
    protected inputNodeCount: number;
    protected activationFunction: any //@TODO
    protected activationFunctionDerivative: any //@TODO
    protected errorGradients: Float64Array;
    protected optimizer: any; //@TODO
    protected inputLayer: any; //@TODO
    protected outputLayer: any; //@TODO

    constructor(nodeCount, activationFunction, optimizer) {
        this.nodeCount = nodeCount;
        this.outputs = new Float64Array(nodeCount);
        this.activationFunction = activationFunction.xToY;
        this.activationFunctionDerivative = activationFunction.yToSlope;
        this.errorGradients = new Float64Array(nodeCount);
        this.optimizer = optimizer;
        this.inputs = new Float64Array(0); //@TODO avoid unneeded empty array?;
        this.inputLayer = null;
        this.inputCount = 0; //@TODO avoid this?
        this.inputNodeCount = 0; //@TODO avoid this?
        this.weights = new Float64Array(0); //@TODO avoid unneeded empty array?
        this.weightErrorGradients = new Float64Array(0); //@TODO avoid unneeded empty array?
    }

    setInputLayer(inputLayer: any) {//@TODO type
        this.inputLayer = inputLayer;
        this.inputCount = inputLayer.nodeCount;
        this.inputNodeCount = this.inputCount + 1;//Add 1 for the bias node
        this.weights = new Float64Array(this.nodeCount * this.inputNodeCount);
        this.weightErrorGradients = new Float64Array(this.nodeCount * this.inputNodeCount);
        for (let weightI = 0, weightLen = this.weights.length; weightI < weightLen; weightI++) {
            this.weights[weightI] = Math.random() - 0.5; //@TODO would a gaussian distribution work better?
        }
        this.optimizer.init(this.weights.length);
    }

    feedForward() {
        this.inputs = this.inputLayer.outputs; //@TODO is this needed?

        //Defining these locally speeds up the loop below by reducing object property access
        let inputNodeCount = this.inputNodeCount;
        let weights = this.weights;
        let inputs = this.inputs;
        let activationFunction = this.activationFunction;
        let outputs = this.outputs;
        let nodeCount = this.nodeCount;
        let inputCount = this.inputCount;

        // Defining these here ideally speeds up the loop below
        let inputI;
        let sum;

        for (let neuronI = 0; neuronI < nodeCount; neuronI++) {
            sum = 0;
            for (inputI = 0; inputI < inputCount; inputI++) {
                sum += inputs[inputI] * weights[neuronI * inputNodeCount + inputI];
            }
            sum += weights[neuronI * inputNodeCount + inputCount];//Bias node that always inputs "1"

            outputs[neuronI] = activationFunction(sum);
        }

        return outputs;
    }

    backPropagateCalculateErrorGradient(targetOutputs: any) {
        //Defining these locally speeds up the loop below by reducing object property access
        let nodeCount = this.nodeCount;
        let errorGradients = this.errorGradients;
        let outputs = this.outputs;
        let activationFunctionDerivative = this.activationFunctionDerivative;
        let inputNodeCount = this.inputNodeCount;
        let inputCount = this.inputCount;
        let inputs = this.inputs;
        let weightErrorGradients = this.weightErrorGradients;

        // Defining these here ideally speeds up the loop below
        let inputI;
        let activationErrorGradient

        for (let neuronI = 0; neuronI < nodeCount; neuronI++) {
            activationErrorGradient = (outputs[neuronI] - targetOutputs[neuronI])
                * activationFunctionDerivative(outputs[neuronI]);
            errorGradients[neuronI] = activationErrorGradient;

            for (inputI = 0; inputI < inputCount; inputI++) {
                weightErrorGradients[neuronI * inputNodeCount + inputI] = inputs[inputI] * activationErrorGradient;
            }
            weightErrorGradients[neuronI * inputNodeCount + inputCount] = activationErrorGradient;//Bias node
        }
    }

    backPropagateOptimize(learningTimeStep: number) {
        this.optimizer.optimizeWeights(this.weights, this.weightErrorGradients, learningTimeStep);
    }
}
