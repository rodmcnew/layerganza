export default class StochasticGradientDescent {
    constructor(learningRate) {
        this.learningRate = learningRate;
        this.calculateUpdate = this.calculateUpdate.bind(this);
    }

    calculateUpdate(weightErrorGradient, weightIndex){
        return -this.learningRate * weightErrorGradient;
    }
}
