import * as tf from "@tensorflow/tfjs";
import  "@tensorflow/tfjs-node";
import data from "./data.json";
import dataTesting from "./dataTesting.json"


const trainingData = tf.tensor2d(data.map(item =>[
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))

const testingData = tf.tensor2d(dataTesting.map(item =>[
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))


const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4],
    activateion: "sigmoid",
    units: 5,
}))
model.add(tf.layers.dense({
    inputShape: [5],
    activateion: "sigmoid",
    units: 3,
}))
model.add(tf.layers.dense({
    inputShape: [4],
    activateion: "sigmoid",
    units: 3,
}))
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06)
})

const startTime = Date.now();
model.fit(trainingData, outputData, {epochs: 100})
.then((history) => {
    console.log(history);
    console.log("Done!", Date.now()-startTime);
})