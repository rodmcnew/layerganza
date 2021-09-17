export default class AdamOptimizer {
    private _m: Float64Array; //@TODO fix naming
    private _v: Float64Array; //@TODO fix naming
    constructor() {
        this.optimizeWeights = this.optimizeWeights.bind(this);
        this._m = new Float64Array(0); //@TODO avoid unneeded array?
        this._v = new Float64Array(0); //@TODO avoid unneeded array?
    }

    init(weightCount) {
        this._m = new Float64Array(weightCount);
        this._v = new Float64Array(weightCount);
    }

    optimizeWeights(weights, weightErrorGradients, learningTimeStep: number) {

        //@TODO allow these to be passed into the constructor
        let learningRate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 0.00000001;

        //Defining these locally speeds up the loop below by reducing object property access
        let m = this._m;
        let v = this._v;
        let oneMinusBeta1 = 1 - beta1;
        let oneMinusBeta2 = 1 - beta2;
        let mtDivisor = (1 - Math.pow(beta1, learningTimeStep));
        let vtDivisor = (1 - Math.pow(beta2, learningTimeStep));

        let gradient;

        for (let i = 0, len = weights.length; i < len; i++) {
            gradient = weightErrorGradients[i];

            m[i] = beta1 * m[i] + oneMinusBeta1 * gradient;
            v[i] = beta2 * v[i] + oneMinusBeta2 * gradient * gradient;
            weights[i] += -learningRate
                * m[i] / mtDivisor
                / (Math.sqrt(v[i] / vtDivisor) + eps);
        }
    }
}
