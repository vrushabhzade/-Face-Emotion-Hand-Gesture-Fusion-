# pyrefly: ignore [missing-import]
import cv2
import time
from collections import deque
import mediapipe as mp
import numpy as np

# Import modules from our project
from gesture_utils import get_hands_detector, get_finger_states, classify_gesture, mp_draw, mp_hands
from emotion_model import get_emotion
from fusion_logic import get_smoothed_state, fuse_states

def main() -> None:
    """
    Main function to run the Face Emotion and Hand Gesture Fusion application.
    Opens the default webcam, runs real-time MediaPipe hand tracking, FER emotion 
    detection (cached every 5 frames), applies temporal smoothing, fuses states,
    and displays a styled GUI overlay with FPS tracking.
    """
    # 1. Initialize Video Capture (Scan indexes 0-3 to find an active webcam)
    cap = None
    for camera_idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f"Successfully opened webcam at index {camera_idx}")
            break
        cap.release()
    else:
        print("Error: Could not open webcam at any index (0-3). Please check your camera connection.")
        return

    # Set frame width and height to a standard size for good performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. Initialize MediaPipe Hands Detector
    hands_detector = get_hands_detector(max_num_hands=2, min_detection_confidence=0.7)

    # 3. Initialize History Deques (temporal smoothing window size = 10)
    emotion_history = deque(maxlen=10)
    gesture_history = deque(maxlen=10)

    # 4. State Cache Variables
    cached_emotion = "Unknown"
    cached_confidence = 0.0
    
    # 5. FPS & Performance Timing Variables
    prev_tick = cv2.getTickCount()
    fps = 0.0
    frame_count = 0

    print("Starting Face Emotion + Hand Gesture Fusion Desktop App...")
    print("Press 'q' to quit.")

    while True:
        # Measure time start for FPS calculation
        loop_start = cv2.getTickCount()

        # Read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Flip the frame horizontally for natural mirror display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ----------------------------------------------------
        # Part A: Hand Gesture Processing (Every frame)
        # ----------------------------------------------------
        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(frame_rgb)

        detected_gesture = "Unknown"
        
        # If hands are detected
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw hand skeleton connections
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2), # Landmark joints (Cyan)
                    mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)                    # Connection lines (Yellowish-green)
                )
                
                # We classify for the primary hand (first hand detected) to feed to fusion logic
                if idx == 0:
                    try:
                        # Extract handedness label: "Left" or "Right"
                        handedness_label = hand_results.multi_handedness[idx].classification[0].label
                        # Calculate finger extension states
                        finger_states = get_finger_states(hand_landmarks, handedness_label)
                        # Classify gesture based on states
                        detected_gesture = classify_gesture(finger_states)
                        
                        # Draw individual gesture text floating above the primary hand
                        x_coord = int(hand_landmarks.landmark[0].x * w)
                        y_coord = int(hand_landmarks.landmark[0].y * h) - 20
                        cv2.putText(
                            frame, 
                            f"{handedness_label}: {detected_gesture}", 
                            (max(10, x_coord), max(20, y_coord)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (255, 255, 0), # Cyan
                            2
                        )
                    except Exception as e:
                        # Fail-safe for individual hand extraction
                        pass
        else:
            detected_gesture = "Unknown"

        # ----------------------------------------------------
        # Part B: Facial Emotion Processing (Every 5th frame)
        # ----------------------------------------------------
        if frame_count % 5 == 0:
            # FER is computationally heavy, so only run periodically
            cached_emotion, cached_confidence = get_emotion(frame)
            
        # ----------------------------------------------------
        # Part C: Temporal Smoothing & State Fusion
        # ----------------------------------------------------
        # Add current detections to rolling histories
        emotion_history.append(cached_emotion)
        gesture_history.append(detected_gesture)

        # Get majority-vote smoothed state
        smoothed_emotion, smoothed_gesture = get_smoothed_state(emotion_history, gesture_history)

        # Get fused action label
        fused_state = fuse_states(smoothed_emotion, smoothed_gesture)

        # ----------------------------------------------------
        # Part D: GUI Overlay & Visualization
        # ----------------------------------------------------
        # Draw a stylish semi-transparent dashboard background panel in the top-left corner
        panel_x1, panel_y1 = 10, 10
        panel_x2, panel_y2 = 340, 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)
        # Apply transparency blend
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        # Draw thin border around panel
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 1)

        # Draw Dashboard Title
        cv2.putText(frame, "FUSION DASHBOARD", (panel_x1 + 15, panel_y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.line(frame, (panel_x1 + 15, panel_y1 + 35), (panel_x2 - 15, panel_y1 + 35), (100, 100, 100), 1)

        # 1. Overlay Emotion (Green color)
        emotion_str = f"Emotion: {smoothed_emotion.capitalize()} ({cached_confidence*100:.1f}%)"
        cv2.putText(frame, emotion_str, (panel_x1 + 15, panel_y1 + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 2. Overlay Gesture (Cyan color)
        gesture_str = f"Gesture: {smoothed_gesture}"
        cv2.putText(frame, gesture_str, (panel_x1 + 15, panel_y1 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # 3. Overlay Fused State (Red for distress alerts, green/white for normal states)
        is_alert = "ALERT" in fused_state
        fused_color = (0, 0, 255) if is_alert else (255, 255, 255) # Red for alerts, White for normal
        
        cv2.putText(frame, f"State: {fused_state}", (panel_x1 + 15, panel_y1 + 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fused_color, 2)

        # 4. Overlay FPS (Gray/White)
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x1 + 15, panel_y1 + 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw a red pulse icon next to "ALERT" in case of danger
        if is_alert:
            # Draw a blinking red circle on the right side of the dashboard
            pulse = int((time.time() * 5) % 2)
            if pulse == 0:
                cv2.circle(frame, (panel_x2 - 30, panel_y1 + 135), 8, (0, 0, 255), -1)

        # Display the frame in the GUI window
        cv2.imshow("Face Emotion & Hand Gesture Fusion", frame)

        # ----------------------------------------------------
        # Part E: Performance Monitoring & Frame Tick Updates
        # ----------------------------------------------------
        frame_count += 1
        
        # Calculate real-time FPS
        loop_end = cv2.getTickCount()
        time_diff = (loop_end - loop_start) / cv2.getTickFrequency()
        
        # Exponential moving average for smoothed FPS
        current_fps = 1.0 / time_diff if time_diff > 0 else 0.0
        fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

        # Check for keyboard inputs
        # Wait 1ms for key press, exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and release webcam resources
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated successfully.")

if __name__ == "__main__":
    main()
