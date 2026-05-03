import cv2
import time
from src.detector import FaceDetector
from src.emotion_analyzer import EmotionAnalyzer
from src.utils import draw_face_box, draw_emotion_bars, draw_no_face, draw_fps, enhance_low_light

from src.logger import EmotionLogger

logger = EmotionLogger(log_path='logs/emotion_log.csv')

def main():
    # initialise face detector and emotion analyzer
    print("Loading models...")
    detector = FaceDetector()
    analyzer = EmotionAnalyzer(weights_path='models/fer_resnet18.pth')
    print("Models loaded. Starting webcam...")

    # open webcam
    # index 0 = default webcam
    # cv2.CAP_DSHOW = DirectShow backend — fixes most Windows webcam issues
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened. Press Q to quit.")

    # FPS tracking
    prev_time = time.time()
    fps = 0

    # emotion result cache
    # we don't run model on every frame — only every 2nd frame
    # this keeps display smooth on CPU
    last_emotion = None
    last_confidence = 0
    last_all_probs = {}
    last_faces = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        # ── run detection and analysis every 2nd frame ──
        if frame_count % 2 == 0:
            frame = enhance_low_light(frame)
            last_faces = detector.detect(frame)

            if len(last_faces) > 0:
                # process first face found
                x, y, w, h = last_faces[0]

                # get cropped 48x48 grayscale face
                face_roi = detector.get_face_roi(frame, x, y, w, h)

                # run emotion analysis
                last_emotion, last_confidence, last_all_probs = analyzer.analyze(face_roi)
                logger.log(last_emotion, last_confidence)
                
        # ── draw results on every frame ──
        if len(last_faces) > 0 and last_emotion is not None:
            x, y, w, h = last_faces[0]

            # draw bounding box and emotion label
            frame = draw_face_box(frame, x, y, w, h, last_emotion, last_confidence)

            # draw probability bars
            if last_all_probs:
                frame = draw_emotion_bars(frame, last_all_probs)
            
        else:
            # no face detected
            frame = draw_no_face(frame)

        # ── calculate and draw FPS ──
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frame = draw_fps(frame, fps)

        # ── display frame ──
        cv2.imshow('Emotion Detector', frame)

        # ── quit on Q key ──
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()