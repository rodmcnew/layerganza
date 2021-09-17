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

//Create the model
let network = new Network(
    [
        new InputLayer(2),
        new HiddenLayer(100, new LeakyRelu(), new AdamOptimizer()),
        new OutputLayer(6, new Linear(), new AdamOptimizer())
    ],
);

const trainingSets: any = []; //@TODO type
for (let i = 0; i < 10; i++) {
    trainingSets.push(
        [
            (new Float64Array(100).map(() => Math.random())),
            (new Float64Array(4).map(() => Math.random()))
        ]
    )
}

while (true) {
    for (let i = 0; i < 3; i++) {
        console.time('t')
        shuffleTrain(network, trainingSets, 10000);
        console.timeEnd('t');
    }
}
//Get some output from the model
//console.log('Output for input [1,1]:', network.invoke([1, 1]));
