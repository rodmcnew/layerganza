export default class StochasticGradientDescent {
    constructor(learningRate) {
        this.learningRate = learningRate;
        this.optimizeWeights = this.optimizeWeights.bind(this);
    }

    optimizeWeights(weights, weightErrorGradients) {
        //Defining locally speeds up the loop below by reducing object property access
        var learningRate = this.learningRate;

        for (var i = 0, len = weights.length; i < len; i++) {
            weights[i] -= learningRate * weightErrorGradients[i];
        }
    }
}
