This is a feed forward neural network with injectable layers, activation functions, and optimizers.

Installation:
```bash
npm install layerganza
```

Example usage:
```js
import DeepNetwork from 'layerganza/lib/DeepNetwork'
import Linear from 'layerganza/lib/activation-function/Linear'
import LeakyRelu from 'layerganza/lib/activation-function/LeakyRelu'
import InputLayer from 'layerganza/lib/layer/InputLayer'
import HiddenLayer from 'layerganza/lib/layer/HiddenLayer'
import OutputLayer from 'layerganza/lib/layer/OutputLayer'
import StochasticGradientDescent from 'layerganza/lib/optimizer/StochasticGradientDescent'
import shuffleTrain from 'layerganza/lib/trainer/shuffleTrain'

//Create the model
var network = new DeepNetwork(
    [
        new InputLayer(2),
        new HiddenLayer(100, new LeakyRelu(), new StochasticGradientDescent(0.1)),
        new OutputLayer(6, new Linear(), new StochasticGradientDescent(0.1))
    ],
);

//Train the model
var trainingSets = [//Outputs: [XOR, OR, AND, GreaterThan, LessThan, NotBoth]
    [[0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 1], [1, 1, 0, 0, 1, 0]],
    [[1, 0], [1, 1, 0, 1, 0, 0]],
    [[1, 1], [0, 1, 1, 0, 0, 0]],
];
shuffleTrain(network, trainingSets, 100);

//Get some output from the model
console.log('Output for input [1,1]:', network.invoke([1, 1]));
```