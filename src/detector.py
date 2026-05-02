# src/detector.py


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os


class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        # download model if not present
        model_path = 'models/blaze_face_short_range.tflite'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe face detection model...")
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
                model_path
            )
            print("Downloaded.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        # convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # wrap in MediaPipe image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # run detection
        results = self.detector.detect(mp_image)

        if not results.detections:
            return []

        faces = []
        h, w = frame.shape[:2]

        for detection in results.detections:
            bbox = detection.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            fw = min(bbox.width, w - x)
            fh = min(bbox.height, h - y)
            faces.append((x, y, fw, fh))

        return faces

    def get_face_roi(self, frame, x, y, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        return face_roi