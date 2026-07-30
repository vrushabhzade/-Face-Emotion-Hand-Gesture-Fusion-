from typing import Tuple
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from fer.fer import FER

# Initialize FER with MTCNN detector for robust face localization
# This initializes the model once when the module is loaded
try:
    detector = FER(mtcnn=True)
except Exception as e:
    # Fallback to default detector in case MTCNN has initialization issues (e.g. library path issues)
    print(f"Warning: Failed to initialize FER with mtcnn=True ({e}). Falling back to default detector.")
    detector = FER(mtcnn=False)

def get_emotion(frame: np.ndarray) -> Tuple[str, float]:
    """
    Detects the dominant facial emotion and its confidence score from an image frame.
    
    If no face is detected, or if the input frame is invalid, it returns ("Unknown", 0.0)
    gracefully without raising exceptions.
    
    Args:
        frame: A numpy ndarray representing the image frame (BGR format from OpenCV).
        
    Returns:
        A tuple of (emotion_label: str, confidence_score: float).
    """
    if frame is None or frame.size == 0:
        return "Unknown", 0.0

    try:
        # top_emotion returns a tuple (dominant_emotion, score) or (None, None)
        emotion, score = detector.top_emotion(frame)
        
        if emotion is None or score is None:
            return "Unknown", 0.0
            
        return str(emotion), float(score)
    except Exception as e:
        # Gracefully catch any internal FER/TensorFlow exceptions (e.g. dimension mismatch)
        # to ensure the desktop app runs robustly without crashing.
        return "Unknown", 0.0
