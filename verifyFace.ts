import * as faceapi from "face-api.js";
import * as canvas from "canvas";
import path from "path";

// patch node-canvas into face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas: Canvas as any, Image: Image as any, ImageData: ImageData as any });

/**
 * Load the necessary face-api.js models from disk.
 */
async function loadModels(modelsPath: string) {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);
}

/**
 * Returns the 128-d face descriptor for the first face found in the image,
 * or null if no face is detected.
 */
async function getDescriptor(
    imagePath: string
): Promise<Float32Array | null> {
    const img = await canvas.loadImage(imagePath);
    const detection = await faceapi
        .detectSingleFace(img as unknown as faceapi.TNetInput)
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
    knownPath: string,
    testPath: string,
    modelsPath = path.resolve(__dirname, "./models"),
    threshold = 0.6
): Promise<boolean> {
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