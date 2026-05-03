from pydantic import BaseModel
from typing import Dict


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    faces_detected: int