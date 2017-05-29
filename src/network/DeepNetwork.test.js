import Linear from '../activation-function/Linear'
import LeakyRelu from '../activation-function/LeakyRelu'
import DeepNetwork from './DeepNetwork'
import InputLayer from '../layer/InputLayer'
import HiddenLayer from '../layer/HiddenLayer'
import OutputLayer from '../layer/OutputLayer'
import StochasticGradientDescent from '../optimizer/StochasticGradientDescent'
import shuffleTrain from '../trainer/shuffleTrain'

var network = new DeepNetwork(
    [
        new InputLayer(2),
        new HiddenLayer(100, new LeakyRelu(), new StochasticGradientDescent(0.1)),
        new OutputLayer(6, new Linear(), new StochasticGradientDescent(0.1))
    ],
);

var trainingSets = [//Outputs: [XOR, OR, AND, GreaterThan, LessThan, NotBoth]
    [[0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 1], [1, 1, 0, 0, 1, 0]],
    [[1, 0], [1, 1, 0, 1, 0, 0]],
    [[1, 1], [0, 1, 1, 0, 0, 0]],
];
shuffleTrain(network, trainingSets, 100);

// console.log('Output for input [1,1]:', network.invoke([1, 1]));
