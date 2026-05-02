<div align='center'>

# Real-Time Emotion Detector

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

Real-time facial emotion detection using OpenCV and EfficientNet. Detects faces via webcam and classifies 7 emotions with live confidence scores — running fully on CPU.
</div>

## 🔍 How It Works

```
Webcam → OpenCV → Haar Cascade → EfficientNet-B0 → Display
```

## ⚙️ Tech Stack

| | Component | Technology |
|---|---|---|
| 👁️ | Computer Vision | OpenCV 4.x |
| 🎯 | Face Detection | Haar Cascade |
| 🧠 | Deep Learning | PyTorch 2.x |
| 😊 | Emotion Model | EfficientNet-B0 via hsemotion |
| 📦 | Package Manager | uv |

## 📁 Project Structure

```
emotion-detector/
├── app.py
├── src/
│   ├── detector.py
│   ├── emotion_analyzer.py
│   └── utils.py
├── models/
│   ├── emotion_model.py
│   └── haarcascade_frontalface_default.xml
└── pyproject.toml
```

## 🚀 Getting Started

```bash
git clone https://github.com/harmandeep2993/emotion-detector.git
cd emotion-detector
uv venv
.venv\Scripts\activate
uv add opencv-python torch torchvision hsemotion "timm==0.9.2" numpy
python app.py
```

Press **Q** to quit.

## 🎭 Emotions Detected

`😠 Angry` `🤢 Disgust` `😨 Fear` `😊 Happy` `😢 Sad` `😲 Surprise` `😐 Neutral`

## 🗺️ Roadmap

- [x] Real-time emotion detection via webcam
- [x] Probability bar chart for all 7 emotions
- [x] FPS counter
- [ ] Upgrade face detection to MediaPipe
- [ ] Multi-face support
- [ ] FastAPI REST endpoint
- [ ] Docker containerisation

## 👤 Author

**Harmandeep Singh** &nbsp;·&nbsp; [GitHub](https://github.com/harmandeep2993) &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/in/yourprofile) &nbsp;·&nbsp; Leipzig, Germany
