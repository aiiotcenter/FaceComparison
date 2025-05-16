// import '@tensorflow/tfjs-node'; // register native tfjs backend
import * as faceapi from 'face-api.js';
import * as canvas from 'canvas';
import path from 'path';
import { fileURLToPath } from 'url';

// recreate __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// patch node-canvas into face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Default models directory
const MODEL_PATH = path.resolve(__dirname, 'models');
let modelsLoaded = false;

async function loadModels(modelsPath = MODEL_PATH) {
    if (modelsLoaded) return;
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);
    modelsLoaded = true;
}

async function getDescriptor(imagePath) {
    const img = await canvas.loadImage(imagePath);
    const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
    return detection?.descriptor ?? null;
}

/**
 * Compare two images and return true if they belong to the same person
 */
export async function isSamePerson(
    knownPath,
    testPath,
    threshold = 0.6
) {
    // Ensure models loaded once
    await loadModels();

    const knownDesc = await getDescriptor(knownPath);
    if (!knownDesc) {
        throw new Error(`No face found in known image: ${knownPath}`);
    }

    const testDesc = await getDescriptor(testPath);
    if (!testDesc) return false;

    const distance = faceapi.euclideanDistance(knownDesc, testDesc);
    return distance < threshold;
}

/**
 * Identify best matching person from a list against a test image
 */
export async function identifyPerson(
    testPath,
    persons,
    threshold = 0.6
) {
    // Ensure models loaded once
    await loadModels();

    const testDesc = await getDescriptor(testPath);
    if (!testDesc) {
        throw new Error(`No face found in test image: ${testPath}`);
    }

    let bestMatch = { name: 'unknown', distance: Infinity };
    for (const person of persons) {
        // resolve known image path relative to project root or absolute
        const kp = path.resolve(__dirname, person.knownPath);
        const knownDesc = await getDescriptor(kp);
        if (!knownDesc) continue;

        const dist = faceapi.euclideanDistance(knownDesc, testDesc);
        if (dist < bestMatch.distance && dist < threshold) {
            bestMatch = { name: person.name, distance: dist };
        }
    }

    return bestMatch;
}
