import { identifyPerson } from "./verifyFace.js";

const persons = [
    { name: "Ibrahim", knownPath: "./known/ame.png" },
    { name: "Mahmoud", knownPath: "./known/mahmoud.png" },
];

async function runIdentification() {
    try {
        const result = await identifyPerson("./test.png", persons);
        console.log("Match result:", result);
    } catch (error) {
        console.error("Error identifying face:", error);
    }
}

runIdentification();
