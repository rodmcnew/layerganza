export default class GradientDescentOptimizer {
    constructor(learningRate) {
        this.learningRate = learningRate;
        this.optimizeWeights = this.optimizeWeights.bind(this);
    }

    init(weightCount) {
        //Do nothing. We don't use weight count in this optimizer.
    }

    optimizeWeights(weights, weightErrorGradients) {
        //Defining locally speeds up the loop below by reducing object property access
        var learningRate = this.learningRate;

        for (var i = 0, len = weights.length; i < len; i++) {
            weights[i] -= learningRate * weightErrorGradients[i];
        }
    }
}
