export default class InputLayer {
    protected nodeCount: string;
    protected outputs: Float64Array;
    constructor(nodeCount) {
        this.nodeCount = nodeCount;
        this.outputs = new Float64Array(nodeCount);//@TODO new array not really used?
    }

    feedForward(inputs) {
        this.outputs = inputs;
    }
}
