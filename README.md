This is a feed forward neural network with injectable layers, activation functions, and optimizers.

Installation:
```bash
npm install layer-oriented-deep-learning-network-js
```

Example usage:
```js
import DeepNetwork from 'layer-oriented-deep-learning-network-js/lib/DeepNetwork'
import Linear from 'layer-oriented-deep-learning-network-js/lib/activation-function/Linear'
import LeakyRelu from 'layer-oriented-deep-learning-network-js/lib/activation-function/LeakyRelu'
import InputLayer from 'layer-oriented-deep-learning-network-js/lib/layer/InputLayer'
import HiddenLayer from 'layer-oriented-deep-learning-network-js/lib/layer/HiddenLayer'
import OutputLayer from 'layer-oriented-deep-learning-network-js/lib/layer/OutputLayer'
import StochasticGradientDescent from 'layer-oriented-deep-learning-network-js/lib/optimizer/StochasticGradientDescent'
import shuffleTrain from 'layer-oriented-deep-learning-network-js/lib/trainer/shuffleTrain'

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