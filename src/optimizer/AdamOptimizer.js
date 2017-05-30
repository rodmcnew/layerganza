export default class AdamOptimizer {
    constructor() {
        this.optimizeWeights = this.optimizeWeights.bind(this);
    }

    init(weightCount) {
        this._m = new Float64Array(weightCount);
        this._v = new Float64Array(weightCount);
        this._t = 0;
    }

    optimizeWeights(weights, weightErrorGradients) {
        this._t += 1;

        //@TODO allow these to be passed into the constructor
        var learningRate = 0.001;
        var beta1 = 0.9;
        var beta2 = 0.999;
        var eps = 0.00000001;

        //Defining these locally speeds up the loop below by reducing object property access
        var m = this._m;
        var v = this._v;
        var t = this._t;
        var oneMinusBeta1 = 1 - beta1;
        var oneMinusBeta2 = 1 - beta2;
        var mtDivisor = (1 - Math.pow(beta1, t));
        var vtDivisor = (1 - Math.pow(beta2, t));

        for (var i = 0, len = weights.length; i < len; i++) {
            var gradient = weightErrorGradients[i];

            m[i] = beta1 * m[i] + oneMinusBeta1 * gradient;
            v[i] = beta2 * v[i] + oneMinusBeta2 * gradient * gradient;
            weights[i] += -learningRate
                * m[i] / mtDivisor
                / (Math.sqrt(v[i] / vtDivisor) + eps);
        }
    }
}
