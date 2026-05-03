<div align="center">

# 🎭 EmoDetector: Real-Time Emotion Detector

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.x-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazeFace-FF6F00?style=flat&logo=google&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

Real-time facial emotion detection system built with OpenCV, MediaPipe, and EfficientNet-B0. Detects faces via webcam and classifies 7 emotions with live confidence scores. Exposes a REST API containerised with Docker.

![Demo](assets/demo.gif)

</div>

## Problem Statement

Understanding human emotional state traditionally requires human observation or self-reporting through surveys and focus groups, which can be slow, expensive, and subject to bias.

This system automates emotion detection passively from a camera feed. No input is required from the person being observed. A camera captures their face, the system classifies their emotional state in real time, and returns a structured response, enabling applications in retail analytics, interactive displays, healthcare monitoring, and driver safety.

## Demo

`😠 Anger` `🤢 Disgust` `😨 Fear` `😊 Happy` `😢 Sad` `😲 Surprise` `😐 Neutral`

![Happy](assets/happy.png)

![Surprise](assets/surprise.png)

## How It Works

```
Webcam / Image
      ↓
MediaPipe BlazeFace — detects face location (x, y, w, h)
      ↓
OpenCV — crops and resizes face to 48x48 grayscale
      ↓
EfficientNet-B0 — classifies emotion (pretrained on AffectNet)
      ↓
Result — emotion label + confidence score + probability bars
```

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | MediaPipe BlazeFace |
| Emotion Model | EfficientNet-B0 (AffectNet, 400k+ images) |
| Video Processing | OpenCV 4.x |
| Deep Learning | PyTorch 2.x |
| REST API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Package Manager | uv |

## Project Structure

```
emotion-detector/
├── app.py                      # webcam real-time loop
├── api/
│   ├── main.py                 # FastAPI endpoints
│   └── schemas.py              # Pydantic response models
├── src/
│   ├── detector.py             # MediaPipe face detection
│   ├── emotion_analyzer.py     # EfficientNet inference
│   ├── utils.py                # drawing + low-light enhancement
│   └── logger.py               # CSV emotion logging
├── models/
│   ├── emotion_model.py        # model architecture
│   └── blaze_face_short_range.tflite
├── Dockerfile
└── pyproject.toml
```

## Getting Started

### Option 1 — Run webcam app locally

Detects emotions in real time from your webcam. Displays a bounding box, emotion label, confidence score, and probability bars on screen.

```bash
git clone https://github.com/harmandeep2993/facial-emotion-recognition.git
cd facial-emotion-recognition

uv venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac / Linux

uv sync

python app.py
```

Press **Q** to quit.

### Option 2 — Run REST API locally

Starts a FastAPI server. Send an image file and receive an emotion JSON response.

```bash
uvicorn api.main:app --reload
```

Open API docs at `http://localhost:8000/docs` and upload any image via the Swagger UI.

### Option 3 — Run API with Docker

No setup needed. One command pulls the image and starts the server.

```bash
docker pull harmandeep2993/emotion-detector
docker run -p 8000:8000 harmandeep2993/emotion-detector
```

API docs available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/analyze` | Upload image, get emotion JSON |

### Example Response

```json
{
  "emotion": "Happy",
  "confidence": 87.3,
  "all_emotions": {
    "Anger": 1.2,
    "Disgust": 0.0,
    "Fear": 0.1,
    "Happy": 87.3,
    "Sad": 0.0,
    "Surprise": 4.1,
    "Neutral": 7.3
  },
  "faces_detected": 1
}
```

## Key Implementation Details

**Low-light enhancement:** CLAHE is applied to each frame before detection to improve face detection in dim conditions.

**Frame skipping:** Emotion analysis runs every 2nd frame with result caching to maintain smooth display on CPU.

**Confidence threshold:** Predictions below 45% confidence are labelled "Uncertain" to avoid misleading results.

**Emotion logging:** Detected emotions are logged to CSV at 1-second intervals for downstream analysis.

**PyTorch 2.6 compatibility:** Patches `torch.load` for the `weights_only` breaking change introduced in PyTorch 2.6.

## Performance

| Hardware | FPS |
|---|---|
| CPU (i5/i7) | 10–20 FPS |
| CPU (i3) | 5–10 FPS |
| GPU | 60+ FPS |

## Limitations

- Single face detection only, with no multi-face support yet implemented
- Requires reasonable lighting, as performance degrades in very low light despite CLAHE enhancement
- Model is trained on AffectNet and may underperform on faces not well represented in that dataset
- No temporal smoothing, so the emotion label can flicker between frames on borderline predictions
- Not validated for production or clinical use

## Roadmap

- [x] Real-time webcam detection
- [x] MediaPipe BlazeFace face detection
- [x] EfficientNet-B0 emotion classification
- [x] Confidence threshold filtering
- [x] Low-light enhancement (CLAHE)
- [x] Emotion history logging to CSV
- [x] FastAPI REST endpoint
- [x] Docker containerisation
- [ ] Multi-face support
- [ ] WebSocket video streaming
- [ ] Deploy to Hugging Face Spaces

## Practical Applications

This system can serve as an integrable service for any application that needs to understand human emotional state from a camera feed. Retail environments can use it to measure customer reactions to products in real time. Driver monitoring systems can combine it with drowsiness detection to flag stressed or distracted drivers, which is an active area of development at German automotive suppliers like Bosch and Continental. In healthcare, it can monitor patients who cannot self-report pain or distress and flag changes for clinical staff automatically.

The REST API and Docker containerisation mean any frontend or application can send an image to `POST /analyze` and receive structured emotion data back, making it ready to plug into a larger product pipeline.

## License

MIT © 2026 Harmandeep Singh
