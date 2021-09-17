import { ActivationFunction } from "../types";

export default class Tanh implements ActivationFunction {
    xToY(x) {
        return Math.tanh(x);
    }

    yToSlope(y) {
        return 1 - y * y;
    }
}
