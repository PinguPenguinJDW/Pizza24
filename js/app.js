// Get references to DOM elements
const webcamElement = document.getElementById('webcam');
const classificationElement = document.getElementById('classification');

// Create an error display element
const errorElement = document.createElement('div');
errorElement.style.color = 'red';
errorElement.style.marginTop = '20px';
errorElement.style.whiteSpace = 'pre-wrap';
document.body.appendChild(errorElement);

function displayError(message) {
    errorElement.innerText += `${message}\n`;
    console.error(message);  // Also log to the console for additional visibility
}

let model;
let labels;

async function loadModel() {
    try {
        displayError("Attempting to load model...");
        const modelUrl = '/assets/model1/model.json';
        
        // Fetch model.json to check if it is accessible
        const modelResponse = await fetch(modelUrl);
        if (!modelResponse.ok) {
            throw new Error(`Unable to fetch model.json file: ${modelResponse.statusText}`);
        }

        // Load the model
        model = await tf.loadGraphModel(modelUrl);
        displayError("Model loaded successfully.");
        classificationElement.innerText = 'Model loaded. Starting camera...';
    } catch (error) {
        displayError(`Error loading model: ${error.message}`);
        console.error(error);  // Log the complete error to the console for more context
    }

    try {
        displayError("Attempting to load metadata...");
        const metadataResponse = await fetch('/assets/model1/metadata.json');
        if (!metadataResponse.ok) {
            throw new Error(`Unable to fetch metadata.json file: ${metadataResponse.statusText}`);
        }
        const metadataJson = await metadataResponse.json();
        labels = metadataJson.labels;
        displayError("Metadata loaded successfully.");
    } catch (error) {
        displayError(`Error loading metadata: ${error.message}`);
    }

    if (model) {
        startCamera();
    } else {
        displayError("Model not loaded, cannot start camera.");
    }
}

async function startCamera() {
    try {
        displayError("Attempting to access webcam...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream;
        webcamElement.addEventListener('loadeddata', classifyFrame);
        displayError("Webcam access successful.");
    } catch (error) {
        displayError(`Error accessing webcam: ${error.message}`);
    }
}

async function classifyFrame() {
    try {
        displayError("Capturing frame from webcam...");

        // Ensure the webcam feed is correctly initialized
        if (webcamElement.videoWidth === 0 || webcamElement.videoHeight === 0) {
            throw new Error("Webcam feed not initialized properly.");
        }

        const webcamImage = tf.browser.fromPixels(webcamElement);
        displayError("Webcam frame captured.");

        displayError("Resizing image...");
        const resizedImage = tf.image.resizeBilinear(webcamImage, [224, 224]);
        displayError("Image resized.");

        displayError("Normalizing image...");
        const normalizedImage = resizedImage.div(255.0).expandDims(0);
        displayError("Image normalized.");

        displayError("Making prediction...");
        const predictions = await model.predict(normalizedImage).data();
        displayError(`Predictions obtained from model: ${predictions}`);

        const highestPredictionIndex = predictions.indexOf(Math.max(...predictions));
        const confidence = predictions[highestPredictionIndex];
        displayError(`Highest prediction index: ${highestPredictionIndex}, Confidence: ${confidence}`);

        if (confidence > 0.85) {
            classificationElement.innerText = `Prediction: ${labels[highestPredictionIndex]} (${(confidence * 100).toFixed(2)}%)`;
        } else {
            classificationElement.innerText = 'Steady camera';
        }

        // Clean up tensors
        webcamImage.dispose();
        resizedImage.dispose();
        normalizedImage.dispose();

        requestAnimationFrame(classifyFrame);
    } catch (error) {
        displayError(`Error during classification: ${error.message}`);
    }
}

// Initialize the app
async function init() {
    displayError("Initializing app...");
    await loadModel();
}

init();
