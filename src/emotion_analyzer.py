import torch
# fix for PyTorch 2.6 — patch torch.load to use weights_only=False
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load

import cv2
import numpy as np
from hsemotion.facial_emotions import HSEmotionRecognizer

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Uncertain']


class EmotionAnalyzer:
    def __init__(self, weights_path=None):
        print("Loading emotion model...")
        self.recognizer = HSEmotionRecognizer(
            model_name='enet_b0_8_best_afew',
            device='cpu'
        )
        print("Emotion model loaded successfully.")

    def analyze(self, face_roi):
        face_bgr = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)

        emotion, scores = self.recognizer.predict_emotions(
            face_bgr,
            logits=False
        )

        confidence = float(max(scores) * 100)

        all_probs = {
            EMOTIONS[i]: round(float(scores[i]) * 100, 1)
            for i in range(len(EMOTIONS))
            if EMOTIONS[i] != "Uncertain"
        }

        if confidence < 45:
            emotion = "Uncertain"

        return emotion, confidence, all_probs