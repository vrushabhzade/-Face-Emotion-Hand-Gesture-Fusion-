from collections import Counter, deque
from typing import Tuple

def get_smoothed_state(emotion_history: deque, gesture_history: deque) -> Tuple[str, str]:
    """
    Computes the stable emotion and gesture states using a majority vote on the histories.
    
    Args:
        emotion_history: A rolling collections.deque containing recent emotion label strings.
        gesture_history: A rolling collections.deque containing recent gesture label strings.
        
    Returns:
        A tuple of (smoothed_emotion: str, smoothed_gesture: str).
    """
    # Smooth emotion history
    if emotion_history:
        smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
    else:
        smoothed_emotion = "Unknown"

    # Smooth gesture history
    if gesture_history:
        smoothed_gesture = Counter(gesture_history).most_common(1)[0][0]
    else:
        smoothed_gesture = "Unknown"

    return smoothed_emotion, smoothed_gesture

def fuse_states(emotion: str, gesture: str) -> str:
    """
    Fuses the smoothed emotion and gesture states based on predefined security/engagement rules.
    
    Fusion rules:
    - (fear / angry / sad) + Fist -> "ALERT: Possible Distress"
    - happy + Open Palm -> "Positive Engagement"
    - neutral / any + Pointing -> "Requesting Attention"
    - Default -> "Normal"
    
    Args:
        emotion: The smoothed facial emotion label.
        gesture: The smoothed hand gesture label.
        
    Returns:
        A fused status message string.
    """
    # Normalize inputs for case-insensitive matching
    emotion_lower = emotion.strip().lower()
    gesture_lower = gesture.strip().lower()

    # Rule 1: Distress Alert
    if emotion_lower in {"fear", "angry", "sad"} and gesture_lower == "fist":
        return "ALERT: Possible Distress"

    # Rule 2: Positive Engagement
    if emotion_lower == "happy" and gesture_lower == "open palm":
        return "Positive Engagement"

    # Rule 3: Requesting Attention
    # "neutral / any + Pointing" -> if gesture is Pointing, return "Requesting Attention"
    if gesture_lower == "pointing":
        return "Requesting Attention"

    # Rule 4: Default fallback
    return "Normal"
