import {
    Network,
    InputLayer,
    HiddenLayer,
    OutputLayer,
    Linear,
    LeakyRelu,
    shuffleTrain,
    AdamOptimizer
} from './index'

const shuffleTrainLog = (epoch: any, setNumber: any, inputs: any, targetOutputs: any, outputs: any) => {
    // console.log('inputs:', inputs);
    let errors = new Array(outputs.length);
    for (let i = 0, len = outputs.length; i < len; i++) {
        errors[i] = Math.abs(targetOutputs[i] - outputs[i]).toFixed(4);
    }
    console.log(
        epoch + ':' + setNumber,
        'errors', errors,
        // 'input', inputs, 'targetOutput', targetOutputs, 'output', outputs
    );
}

//Create the model
let network = new Network(
    [
        new InputLayer(2),
        new HiddenLayer(100, new LeakyRelu(), new AdamOptimizer()),
        new OutputLayer(6, new Linear(), new AdamOptimizer())
    ],
);

//Train the model
let trainingSets = [//Outputs: [XOR, OR, AND, GreaterThan, LessThan, NotBoth]
    [[0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 1], [1, 1, 0, 0, 1, 0]],
    [[1, 0], [1, 1, 0, 1, 0, 0]],
    [[1, 1], [0, 1, 1, 0, 0, 0]],
];
shuffleTrain(network, trainingSets, 400, shuffleTrainLog);

//Get some output from the model
//console.log('Output for input [1,1]:', network.invoke([1, 1]));
