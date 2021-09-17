import arrayShuffle from '../math/arrayShuffle'

type LogFunction = (epoc, setI, set0, set1, outputs) => void; //@TODO

export default function shuffleTrain(neuralNetwork: any, trainingSets: any, maxEpochs: any, log: LogFunction = () => { }) {
    for (let epoch = 0; epoch < maxEpochs; epoch++) {
        trainingSets = arrayShuffle(trainingSets);

        for (let setI = 0, setCount = trainingSets.length; setI < setCount; setI++) {

            let set = trainingSets[setI];

            let outputs = neuralNetwork.invoke(set[0]);

            neuralNetwork.learn(set[1]);

            log(epoch, setI, set[0], set[1], outputs);
        }
    }
}
