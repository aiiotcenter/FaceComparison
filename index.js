import express from "express";
import multer from "multer";
import { identifyPerson } from "./verifyFace.js";
import fs from "fs";

const app = express();
const port = 3000;

// Configure multer for file uploads
const upload = multer({ dest: "uploads/" });

const persons = [
    { name: "Ibrahim", knownPath: "./known/ame.png" },
    { name: "Mahmoud", knownPath: "./known/mahmoud.png" },
];

// POST /detectface endpoint
app.post("/detectface", upload.single("image"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No image uploaded" });
        }
        const result = await identifyPerson(req.file.path, persons);

        // Optionally delete the uploaded file after processing
        fs.unlink(req.file.path, () => {});

        res.json({ match: result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
