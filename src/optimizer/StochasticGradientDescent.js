export default class StochasticGradientDescent {
    constructor(learningRate) {
        this.learningRate = learningRate;
        this.optimizeWeights = this.optimizeWeights.bind(this);
    }

    optimizeWeights(weights, weightErrorGradients) {
        for (var i = 0, len = weights.length; i < len; i++) {
            weights[i] -= this.learningRate * weightErrorGradients[i];
        }
    }
}
