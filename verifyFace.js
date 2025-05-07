import '@tensorflow/tfjs-node';  // If needed, uncomment this line
import * as faceapi from "face-api.js";
import * as canvas from "canvas";
import path from "path";
import { fileURLToPath } from "url";

// define __dirname for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// patch node-canvas into face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

/**
 * Load the necessary face-api.js models from disk.
 */
async function loadModels(modelsPath) {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);
}

/**
 * Returns the 128-d face descriptor for the first face found in the image,
 * or null if no face is detected.
 */
async function getDescriptor(imagePath) {
    const img = await canvas.loadImage(imagePath);
    const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
    return detection?.descriptor ?? null;
}

/**
 * Compares two images and returns true if they’re the same person.
 *
 * @param knownPath   Path to the “known” image of your one person.
 * @param testPath    Path to the image you want to verify.
 * @param modelsPath  Directory where you’ve put the face-api.js models.
 * @param threshold   Max Euclidean distance for a “match” (default 0.6).
 */
export async function isSamePerson(
    knownPath,
    testPath,
    modelsPath = path.resolve(__dirname, "./models"),
    threshold = 0.6
) {
    // Load models once
    await loadModels(modelsPath);

    // Get descriptors
    const knownDesc = await getDescriptor(knownPath);
    if (!knownDesc)
        throw new Error(`No face found in known image: ${knownPath}`);

    const testDesc = await getDescriptor(testPath);
    if (!testDesc) return false; // no face → definitely not the same

    // Compare distance
    const distance = faceapi.euclideanDistance(knownDesc, testDesc);
    return distance < threshold;
}

/**
 * Identifies which person a test image belongs to.
 *
 * @param testPath  Path to the image you want to verify.
 * @param persons   Array of person objects in the format:
 *                  [{ name: "Alice", knownPath: "/path/to/alice.jpg" }, ...]
 * @param modelsPath Directory where you’ve put the face-api.js models.
 * @param threshold Max Euclidean distance for a “match” (default 0.6).
 *
 * @returns An object { name: string, distance: number }.
 *          If no match is found, name will be "unknown".
 */
export async function identifyPerson(
    testPath,
    persons,
    modelsPath = path.resolve(__dirname, "./models"),
    threshold = 0.6
) {
    // Load models once
    await loadModels(modelsPath);

    // Get descriptor for the test image
    const testDesc = await getDescriptor(testPath);
    if (!testDesc)
        throw new Error(`No face found in test image: ${testPath}`);

    let bestMatch = { name: "unknown", distance: Infinity };

    // Iterate over persons and compare against test image descriptor
    for (const person of persons) {
        const knownDesc = await getDescriptor(person.knownPath);
        if (!knownDesc) continue; // Skip if the known image failed
        const distance = faceapi.euclideanDistance(knownDesc, testDesc);
        if (distance < bestMatch.distance && distance < threshold) {
            bestMatch = { name: person.name, distance };
        }
    }

    return bestMatch;
}
