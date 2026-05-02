# src/detector.py


import cv2


class FaceDetector:
    def __init__(self, cascade_path='models/haarcascade_frontalface_default.xml'):
        # load the haar cascade from the xml file
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")

    def detect(self, frame):
        # convert frame to grayscale
        # haar cascade works on grayscale images only
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the frame
        # scaleFactor  — how much the image is scaled down each pass
        #                1.1 means 10% reduction each time
        # minNeighbors — how many overlapping detections needed to confirm a face
        #                higher = fewer false positives
        # minSize      — minimum face size to detect (pixels)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        # if no faces found — return empty list
        if len(faces) == 0:
            return []

        return faces

    def get_face_roi(self, frame, x, y, w, h):
        # ROI = Region of Interest
        # crop just the face from the full frame
        # convert to grayscale for the emotion model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]

        # resize to 48x48 — the size FER2013 was trained on
        face_roi = cv2.resize(face_roi, (48, 48))

        return face_roi