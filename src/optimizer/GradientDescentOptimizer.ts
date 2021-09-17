//@TODO have a default learning rate
export default class GradientDescentOptimizer {
    protected learningRate: number;
    constructor(learningRate) {
        this.learningRate = learningRate;
        this.optimizeWeights = this.optimizeWeights.bind(this);
    }

    init() {
        //Do nothing. We don't use weight count in this optimizer.
    }

    optimizeWeights(weights, weightErrorGradients) {
        //Defining locally speeds up the loop below by reducing object property access
        let learningRate = this.learningRate;

        for (let i = 0, len = weights.length; i < len; i++) {
            weights[i] -= learningRate * weightErrorGradients[i];
        }
    }
}
