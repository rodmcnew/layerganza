import {
    Network,
    InputLayer,
    HiddenLayer,
    OutputLayer,
    Linear,
    LeakyRelu,
    AdamOptimizer,
    shuffleTrain
} from './index'

//Create the model
var network = new Network(
    [
        new InputLayer(2),
        new HiddenLayer(100, new LeakyRelu(), new AdamOptimizer()),
        new OutputLayer(6, new Linear(), new AdamOptimizer())
    ],
);

//Train the model
var trainingSets = [//Outputs: [XOR, OR, AND, GreaterThan, LessThan, NotBoth]
    [[0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 1], [1, 1, 0, 0, 1, 0]],
    [[1, 0], [1, 1, 0, 1, 0, 0]],
    [[1, 1], [0, 1, 1, 0, 0, 0]],
];
shuffleTrain(network, trainingSets, 300);

//Get some output from the model
//console.log('Output for input [1,1]:', network.invoke([1, 1]));
