import { ActivationFunction } from "../types";

export default class LogisticSigmoid implements ActivationFunction {
    xToY(x) {
        return 1 / (1 + Math.exp(-x));
    }

    yToSlope(y) {
        return y * (1 - y);
    }
}
