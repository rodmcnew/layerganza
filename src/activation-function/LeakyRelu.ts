import { ActivationFunction } from "../types";

const slopeBelowZero = 0.01;
export default class LeakyRelu implements ActivationFunction {
    xToY(x) {
        if (x > 0) {
            return x
        } else {
            return slopeBelowZero * x
        }
    }

    yToSlope(y) {
        if (y > 0) {
            return 1
        } else {
            return slopeBelowZero;
        }
    }
}
