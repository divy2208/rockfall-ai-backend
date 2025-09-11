
import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import twilio from "twilio";
import fetch from "node-fetch";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Twilio client
const client = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);

// Health check
app.get("/", (req, res) => {
  res.send("🚀 Backend is running");
});

// --------------------
// Twilio SMS Route
// --------------------
app.post("/api/sms", async (req, res) => {
  const { to, body } = req.body;

  if (!to || !body) {
    return res.status(400).json({ error: "Phone number and message are required" });
  }

  try {
    const message = await client.messages.create({
      body,
      from: process.env.TWILIO_PHONE,
      to,
    });
    res.json({ success: true, sid: message.sid });
  } catch (err) {
    console.error("Twilio Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// --------------------
// Hugging Face Risk Analysis Route
// --------------------
app.post("/api/analyze", async (req, res) => {
  try {
    const { text } = req.body;

    const response = await fetch(
      "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.HF_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: text,
          parameters: { candidate_labels: ["High Risk", "Moderate Risk", "Safe"] },
        }),
      }
    );

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Hugging Face Error:", err);
    res.status(500).json({ error: "Failed to analyze risk" });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Backend running on http://localhost:${PORT}`));
