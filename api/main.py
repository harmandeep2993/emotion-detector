import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from api.schemas import EmotionResponse
from src.detector import FaceDetector
from src.emotion_analyzer import EmotionAnalyzer

app = FastAPI(
    title="Emotion Detector API",
    description="Real-time facial emotion detection using MediaPipe and EfficientNet-B0",
    version="1.0.0"
)

# load models once at startup — not on every request
print("Loading models...")
detector = FaceDetector()
analyzer = EmotionAnalyzer()
print("Models ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B0", "detector": "MediaPipe BlazeFace"}


@app.post("/analyze", response_model=EmotionResponse)
async def analyze(file: UploadFile = File(...)):
    # validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG supported."
        )

    # read uploaded bytes
    image_bytes = await file.read()

    # convert bytes → PIL Image → NumPy array
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(pil_image)

    # convert RGB → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # detect faces
    faces = detector.detect(frame)

    if len(faces) == 0:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image."
        )

    # analyse first face
    x, y, w, h = faces[0]
    face_roi = detector.get_face_roi(frame, x, y, w, h)
    emotion, confidence, all_probs = analyzer.analyze(face_roi)

    return EmotionResponse(
        emotion=emotion,
        confidence=round(confidence, 1),
        all_emotions=all_probs,
        faces_detected=len(faces)
    )