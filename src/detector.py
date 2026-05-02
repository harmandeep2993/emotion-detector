import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        # initialise MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils

        # min_detection_confidence — how confident MediaPipe must be
        # before reporting a face (0.0 to 1.0)
        # 0.6 = 60% confidence threshold — good balance
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,              # 0 = short range (within 2m) — good for webcam
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        # MediaPipe requires RGB — OpenCV uses BGR
        # convert before passing to detector
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run detection
        results = self.detector.process(rgb_frame)

        # if no faces found return empty list
        if not results.detections:
            return []

        # convert MediaPipe results to (x, y, w, h) format
        # same format as Haar Cascade — so rest of code stays unchanged
        faces = []
        h, w = frame.shape[:2]

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # MediaPipe returns relative coordinates (0.0 to 1.0)
            # multiply by frame dimensions to get pixel coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            fw = int(bbox.width * w)
            fh = int(bbox.height * h)

            # clamp to frame boundaries — prevent negative coordinates
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            faces.append((x, y, fw, fh))

        return faces

    def get_face_roi(self, frame, x, y, w, h):
        # crop face from frame
        # convert to grayscale for emotion model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]

        # resize to 48x48 — input size emotion model expects
        face_roi = cv2.resize(face_roi, (48, 48))

        return face_roi