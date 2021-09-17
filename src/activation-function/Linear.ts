import { ActivationFunction } from "../types";

export default class Linear implements ActivationFunction {
    xToY(x) {
        return x;
    }

    yToSlope() {
        return 1
    }
}
