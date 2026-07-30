from typing import List
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create a default instance configuration to be imported/used in main.py
def get_hands_detector(max_num_hands: int = 2, min_detection_confidence: float = 0.7) -> mp.solutions.hands.Hands:
    """
    Initializes and returns a MediaPipe Hands detector instance.
    
    Args:
        max_num_hands: Maximum number of hands to detect.
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection.
        
    Returns:
        An instance of mp.solutions.hands.Hands.
    """
    return mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5
    )

def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
    """
    Determines the state (extended or curled) of each of the 5 fingers.
    
    The finger states list format: [thumb, index, middle, ring, pinky]
    where 1 = finger extended, 0 = curled.
    
    Args:
        hand_landmarks: The MediaPipe hand landmarks object containing 21 landmarks.
        handedness_label: The anatomical handedness label ("Left" or "Right").
        
    Returns:
        A 5-bit list of integers (0 or 1) representing the finger states.
    """
    landmarks = hand_landmarks.landmark
    states = [0] * 5

    # 1. Thumb State (compare landmark 4 vs 3, adjusted for left/right hand)
    # MediaPipe handedness label corresponds to the anatomical hand.
    # In standard camera view (non-mirrored or mirrored), MediaPipe correctly identifies
    # Left vs Right.
    # For anatomical Left hand, thumb extends to the right (x increasing) relative to IP joint.
    # For anatomical Right hand, thumb extends to the left (x decreasing) relative to IP joint.
    if handedness_label == "Left":
        states[0] = 1 if landmarks[4].x > landmarks[3].x else 0
    else:  # "Right"
        states[0] = 1 if landmarks[4].x < landmarks[3].x else 0

    # 2. Other fingers (compare tip landmark y vs pip landmark y)
    # y-coordinate increases downwards in MediaPipe coordinate system.
    # Tip is extended if its y-coordinate is smaller (higher up) than pip.
    # Index: tip (8) vs pip (6)
    states[1] = 1 if landmarks[8].y < landmarks[6].y else 0
    # Middle: tip (12) vs pip (10)
    states[2] = 1 if landmarks[12].y < landmarks[10].y else 0
    # Ring: tip (16) vs pip (14)
    states[3] = 1 if landmarks[16].y < landmarks[14].y else 0
    # Pinky: tip (20) vs pip (18)
    states[4] = 1 if landmarks[20].y < landmarks[18].y else 0

    return states

def classify_gesture(states: List[int]) -> str:
    """
    Classifies the gesture label based on the 5-finger bit pattern list.
    
    Args:
        states: A list of 5 integers (0 or 1) for [thumb, index, middle, ring, pinky].
        
    Returns:
        A string label representing the classified gesture:
        "Fist", "Open Palm", "Peace", "Thumbs Up", "Pointing", "Call Me", or "Unknown".
    """
    # Convert list of finger states to a tuple for hashable dictionary lookup
    state_tuple = tuple(states)
    
    gesture_map = {
        (0, 0, 0, 0, 0): "Fist",
        (1, 1, 1, 1, 1): "Open Palm",
        (0, 1, 1, 0, 0): "Peace",
        (1, 0, 0, 0, 0): "Thumbs Up",
        (0, 1, 0, 0, 0): "Pointing",
        (1, 0, 0, 0, 1): "Call Me"
    }
    
    return gesture_map.get(state_tuple, "Unknown")
