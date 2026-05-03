# src/logger.py

import csv
import os
import time
from datetime import datetime

class EmotionLogger:
    def __init__(self, log_path='logs/emotion_log.csv'):
        self.log_path = log_path
        self.last_log_time = 0
        self.log_interval = 1.0  # log once per second maximum

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'emotion', 'confidence'])

    def log(self, emotion, confidence):
        if emotion == "Uncertain":
            return

        # only log once per second
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return

        self.last_log_time = current_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, round(confidence, 1)])