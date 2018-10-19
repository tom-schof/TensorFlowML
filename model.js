const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const data = require("./data.json");
const dataTesting = require("./dataTesting.json");
// const data256 = "./256_ObjectCategories";
// const fs = require("fs");

const trainingData = tf.tensor2d(data.map(item =>[
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));
const outputData = tf.tensor2d(data.map(item=> [
    item.species === "setosa" ? 1:0,
    item.species === "virginica" ? 1:0,
    item.species === "versicolor" ? 1:0,

]));

const testingData = tf.tensor2d(dataTesting.map(item =>[
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));


const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 100,
}));
model.add(tf.layers.dense({
    inputShape: [100],
    activation: "sigmoid",
    units: 50,
}));
model.add(tf.layers.dense({
    inputShape: [50],
    activation: "sigmoid",
    units: 3,
}));
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.05)
});

const startTime = Date.now();
model.fit(trainingData, outputData, {epochs: 300})
.then((history) => {
    console.log(history);
    console.log("Done! Trained in", ((Date.now()-startTime)/1000), "seconds.");
    model.predict(testingData).print();
});


// fs.readdir(data256, function (err, items) {
//     // console.log(items);
//     for (let i = 0; i < 1; i++) {
//         let folder = data256 + '/' + items[i];
//         console.log(folder);
//         fs.readdir(folder, function (err, folderItems) {
//             // console.log(folder);
//             // console.log(folderItems);
//             for (let j = 0; j < folderItems.length; j++) {
//                 let file = folder + '/' + folderItems[j];
//                 // console.log(file);
//                 // loadAndProcessImage(file);
//             }
//         });
//     }
// });

// function loadImage(src) {
//     return new Promise((resolve, reject) => {
//         const img = new Image();
//         console.log(img);
//         console.log(src);
//         img.src = src;
//         img.onload = () => resolve(tf.fromPixels(img));
//         img.onerror = (err) => reject(err);
//     });
// }

// function cropImage(img) {
//     console.log(img);
//     const width = img.shape[0];
//     const height = img.shape[1];

//     // use the shorter side as the size to which we will crop
//     const shorterSide = Math.min(img.shape[0], img.shape[1]);

//     // calculate beginning and ending crop points
//     const startingHeight = (height - shorterSide) / 2;
//     const startingWidth = (width - shorterSide) / 2;
//     const endingHeight = startingHeight + shorterSide;
//     const endingWidth = startingWidth + shorterSide;

//     // return image data cropped to those points
//     return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
// }

// function resizeImage(image) {
//     return tf.image.resizeBilinear(image, [224, 224]);
// }

// function batchImage(image) {
//     // Expand our tensor to have an additional dimension, whose size is 1
//     const batchedImage = image.expandDims(0);

//     // Turn pixel data into a float between -1 and 1.
//     return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
// }

// function loadAndProcessImage(src) {
//     const image = loadImage(src);
//     const croppedImage = cropImage(image);
//     console.log("cropped image:" + croppedImage);
//     const resizedImage = resizeImage(croppedImage);
//     console.log("resized image:" + resizedImage);
//     const batchedImage = batchImage(resizedImage);
//     console.log(batchedImage);
//     return batchedImage;
// }

// loadImage("./256_ObjectCategories/001.ak47/001_0001.jpg");
// loadAndProcessImage("./256_ObjectCategories/001.ak47/001_0001.jpg");
// const trainingData = tf.tensor2d(data256.map(item =>[
//     item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
// ]));
// const outputData = tf.tensor2d(data.map(item=> [
//     item.species === "setosa" ? 1:0,
//     item.species === "virginica" ? 1:0,
//     item.species === "versicolor" ? 1:0,

// ]));

// const testingData = tf.tensor2d(dataTesting.map(item =>[
//     item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
// ]));


// const model = tf.sequential();

// model.add(tf.layers.dense({
//     inputShape: [4],
//     activation: "relu",
//     units: 4000,
// }));
// model.add(tf.layers.dense({
//     inputShape: [100],
//     activation: "relu",
//     units: 2000,
// }));
// model.add(tf.layers.dense({
//     inputShape: [50],
//     activation: "sigmoid",
//     units: 3,
// }));
// model.compile({
//     loss: "meanSquaredError",
//     optimizer: tf.train.adam(.05)
// });

// const startTime = Date.now();
// model.fit(trainingData, outputData, {epochs: 300})
// .then((history) => {
//     console.log(history);
//     console.log("Done! Trained in", ((Date.now()-startTime)/1000), "seconds.");
//     model.predict(testingData).print();
// });




