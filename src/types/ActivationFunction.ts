export interface ActivationFunction {
    xToY: (x: number) => number;
    yToSlope: (y: number) => number;
}
