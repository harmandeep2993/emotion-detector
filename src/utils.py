import cv2

# emotion to colour mapping
# each emotion gets a distinct colour
# colours are in BGR format (OpenCV uses BGR not RGB)
EMOTION_COLORS = {
    'Angry':    (0, 0, 255),      # red
    'Disgust':  (0, 140, 255),    # orange
    'Fear':     (0, 255, 255),    # yellow
    'Happy':    (0, 255, 0),      # green
    'Sad':      (255, 0, 0),      # blue
    'Surprise': (255, 0, 255),    # pink
    'Neutral':  (255, 255, 255),  # white
    'Uncertain': (128, 128, 128), # gray
}


def draw_face_box(frame, x, y, w, h, emotion, confidence):
    # get colour for this emotion
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # draw rectangle around face
    cv2.rectangle(
        frame,
        (x, y),          # top left corner
        (x + w, y + h),  # bottom right corner
        color,            # colour
        2                 # thickness
    )

    # build label text
    label = f"{emotion} {confidence:.1f}%"

    # draw filled rectangle behind text so it is readable
    # calculate text size first
    (text_w, text_h), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,   # font scale
        2      # thickness
    )

    # filled rectangle sits above the face box
    cv2.rectangle(
        frame,
        (x, y - text_h - 10),
        (x + text_w, y),
        color,
        -1    # -1 means filled
    )

    # draw text on top of filled rectangle
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,           # font scale
        (0, 0, 0),     # black text
        2,             # thickness
        cv2.LINE_AA    # anti-aliased — smoother text
    )

    return frame


def draw_emotion_bars(frame, all_probs):
    # draws a small probability bar chart in top-right corner
    # shows confidence for all 7 emotions at once

    bar_x = frame.shape[1] - 200   # start 200px from right edge
    bar_y = 20                      # start 20px from top
    bar_height = 14
    bar_gap = 18
    max_bar_width = 150

    for i, (emotion, prob) in enumerate(all_probs.items()):
        y_pos = bar_y + (i * bar_gap)
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # draw emotion label
        cv2.putText(
            frame,
            f"{emotion[:3]}",          # first 3 letters to save space
            (bar_x - 35, y_pos + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

        # draw background bar (gray)
        cv2.rectangle(
            frame,
            (bar_x, y_pos),
            (bar_x + max_bar_width, y_pos + bar_height),
            (80, 80, 80),
            -1
        )

        # draw filled bar proportional to probability
        filled_width = int((prob / 100) * max_bar_width)
        if filled_width > 0:
            cv2.rectangle(
                frame,
                (bar_x, y_pos),
                (bar_x + filled_width, y_pos + bar_height),
                color,
                -1
            )

        # draw percentage text
        cv2.putText(
            frame,
            f"{prob:.0f}%",
            (bar_x + max_bar_width + 5, y_pos + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_no_face(frame):
    # display message when no face detected
    cv2.putText(
        frame,
        "No face detected",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    return frame


def draw_fps(frame, fps):
    # display FPS counter top left
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, frame.shape[0] - 20),   # bottom left
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    return frame


def enhance_low_light(frame):
    # convert to LAB colour space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # apply CLAHE — Contrast Limited Adaptive Histogram Equalisation
    # boosts local contrast without overexposing bright areas
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # merge back and convert to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced