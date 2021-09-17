This is a feed forward neural network with injectable layers, activation functions, and optimizers.

Installation:
```bash
npm install layerganza
```

Example usage:
```js
import {
    Network,
    InputLayer,
    HiddenLayer,
    OutputLayer,
    Linear,
    LeakyRelu,
    AdamOptimizer,
    shuffleTrain
} from 'layerganza'

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
shuffleTrain(network, trainingSets, 200);

//Get some output from the model
console.log('Output for input [1,1]:', network.invoke([1, 1]));
```
