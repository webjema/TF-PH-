async function getHealthData() {
  const healthDataReq = await fetch('healthData.json');
  const healthData = await healthDataReq.json();
  const cleanedHealthData = healthData.map(d => ({
    featureA: d.A,
    featureB: d.B,
    label: d.Class
  })).filter(d => (d.featureA != null && d.featureB != null && d.label != null));
  return cleanedHealthData;
}

async function getTestData() {
  const testDataReq = await fetch('testData.json');
  const testData = await testDataReq.json();
  const cleanedTestData = testData.map(d => ({
    featureA: d.A,
    featureB: d.B
  })).filter(d => (d.featureA != null && d.featureB != null));
  return cleanedTestData;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();
  // Add an input layer
  model.add(tf.layers.dense({ inputShape: [2], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 15, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'relu' }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  return model;
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError, //categorical_crossentropy? how?
    metrics: ['acc'],
  });
  const batchSize = 10;
  const epochs = 14;
  const oneHot = tf.oneHot(labels, 3);
  console.log("Train input:"); inputs.print();
  console.log("Labels oneHot:"); oneHot.print(); // debug
  return await model.fit(inputs, oneHot, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'label'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, min, max) {
  const { inputs, labels } = inputData;
  const unNormInput = inputs
    .mul(max.sub(min))
    .add(min);
  console.log("Test data:");unNormInput.print(); // debug
  const preds = model.predict(inputs);
  console.log("Predict:"); preds.print(); // debug
  decodedPred = tf.argMax(preds, axis=1);
  console.log("Decoded Predict:"); decodedPred.print(); // debug
  const decodedPredArray = decodedPred.arraySync();

  // show output data table
  const headers = ['Feature A', 'Feature B', 'Pred-Class'];
  const values = unNormInput.arraySync().map((e, i) => e.concat(decodedPredArray[i]));
  const surface = { name: 'Output health data table', tab: 'Data analisys' };
  tfvis.render.table(surface, { headers, values });
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const healthData = await getHealthData();
  const testData = await getTestData();
  const { min, max } = getMinMax(healthData, testData);
  // show input data table
  const headers = ['Feature A', 'Feature B', 'Class'];
  const values = healthData.map(d => [d.featureA, d.featureB, d.label]);
  const surface = { name: 'Input health data table', tab: 'Data analisys' };
  tfvis.render.table(surface, { headers, values });
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);
  // Convert the data to a form we can use for training.
  const { inputs, labels } = convertToTensor(healthData, min, max);
  // Train the model  
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  // Test model
  const testTensorData = convertToTensor(testData, min, max);
  testModel(model, testTensorData, min, max);
  console.log('Done Testing');
}

document.addEventListener('DOMContentLoaded', run);

/**
* Convert the input data to tensors that we can use for machine 
* learning. We will also do the important best practices of _shuffling_
* the data and _normalizing_ the data
*/
function convertToTensor(data, min, max) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  // Step 1. Shuffle the data    
  tf.util.shuffle(data);
  // Step 2. Convert data to Tensor
  const inputs = data.map(d => [d.featureA, d.featureB])
  const labels = data.map(d => d.label);
  const inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
  const normalizedInputs = inputTensor.sub(min).div(max.sub(min));
  return {
    inputs: normalizedInputs,
    labels: labels
  }
}

function getMinMax(healthData, testData) {
  const inputs1 = healthData.map(d => [d.featureA, d.featureB])
  const inputs2 = testData.map(d => [d.featureA, d.featureB])
  const all = inputs1.concat(inputs2);
  const inputTensor = tf.tensor2d(all, [all.length, 2]);
  const inputMax = inputTensor.max();
  const inputMin = inputTensor.min();
  return { min: inputMin, max: inputMax }
}