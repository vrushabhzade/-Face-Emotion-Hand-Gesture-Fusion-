# Face Emotion + Hand Gesture Fusion Desktop App

A real-time Python desktop application that fuses facial emotion recognition (via the `fer` library using MTCNN face detection) and hand gesture classification (via `mediapipe` hand tracking). The application performs temporal smoothing (rolling window of 10 frames) and applies decision rules to classify user states into "ALERT: Possible Distress", "Positive Engagement", "Requesting Attention", or "Normal".

## Project Structure

```text
face-gesture-fusion/
├── main.py              # Application entry point, OpenCV loop, and dashboard visualization
├── gesture_utils.py     # MediaPipe Hands detector setup, finger states, and gesture mapping
├── emotion_model.py     # FER library wrapper with MTCNN support and safety fallbacks
├── fusion_logic.py      # Rolling window majority voting and state fusion rules
├── requirements.txt     # Python libraries needed to run the application
└── README.md            # Setup, explanation of logic, and roadmap notes
```

## Setup Instructions

Ensure you have Python 3.9 to 3.11 installed. Newer versions may have compatibility issues with tensorflow.

1. **Clone or Navigate to the Directory**:
   ```bash
   cd "c:/Users/VRUSHABH/OneDrive/Music/Desktop/emotion+hand gesture project/face-gesture-fusion"
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **On Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **On Windows (Command Prompt)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## How to Run

Start the application with:
```bash
python main.py
```
- A webcam window titled **"Face Emotion & Hand Gesture Fusion"** will open.
- The dashboard overlay in the top-left displays your current smoothed emotion, hand gesture, fusion state, and real-time FPS.
- Press **'q'** in the window to quit and release resources cleanly.

---

## Finger-State Bit-Vector Logic

Gestures are classified using a **5-bit vector** representing the state of each finger:
`[thumb, index, middle, ring, pinky]`
where `1` means the finger is **extended** and `0` means the finger is **curled**.

```text
Landmarks Index Map:
  Thumb:  [1, 2, 3, 4]  (Tip is 4, IP is 3)
  Index:  [5, 6, 7, 8]  (Tip is 8, PIP is 6)
  Middle: [9, 10, 11, 12] (Tip is 12, PIP is 10)
  Ring:   [13, 14, 15, 16] (Tip is 16, PIP is 14)
  Pinky:  [17, 18, 19, 20] (Tip is 20, PIP is 18)
```

### 1. Index, Middle, Ring, Pinky Extension Rule
For these 4 fingers, we compare the y-coordinate of the **Tip landmark** with the y-coordinate of the **PIP (Proximal Interphalangeal) landmark**.
- In OpenCV/MediaPipe, the y-axis goes **downwards** (0 at the top, 1 at the bottom).
- If `Tip.y < PIP.y`, the tip is higher than the joint, indicating the finger is **extended** (`1`).
- Otherwise, it is **curled** (`0`).

### 2. Thumb Extension Rule
The thumb moves horizontally rather than vertically relative to the palm's center. We compare the x-coordinates of the **Thumb Tip (4)** and **Thumb IP (3)**. We must adjust this based on anatomical hand labels (reported by MediaPipe Handedness classification):
- **Anatomical Left Hand**: The thumb extends outward to the right. Therefore, the thumb is extended (`1`) if `Tip.x > IP.x`.
- **Anatomical Right Hand**: The thumb extends outward to the left. Therefore, the thumb is extended (`1`) if `Tip.x < IP.x`.

### 3. Gesture Classification Mappings
We match the resulting 5-bit tuple against a dictionary of predefined gestures:
- **`[0, 0, 0, 0, 0]`** -> **Fist**
- **`[1, 1, 1, 1, 1]`** -> **Open Palm**
- **`[0, 1, 1, 0, 0]`** -> **Peace**
- **`[1, 0, 0, 0, 0]`** -> **Thumbs Up**
- **`[0, 1, 0, 0, 0]`** -> **Pointing**
- **`[1, 0, 0, 0, 1]`** -> **Call Me**
- Any other pattern -> **Unknown**

---

## State Fusion Logic Rules

To prevent noise and flickering, we maintain a rolling deque of size 10 for both emotion and gesture. We take the **majority vote** (statistical mode) over the last 10 frames to compute a stable state.

These stable states are fused using the following safety/engagement rules:

| Emotion (Smoothed) | Gesture (Smoothed) | Fused Action Label |
|:---|:---|:---|
| `fear` or `angry` or `sad` | `Fist` | **`ALERT: Possible Distress`** (Red blinker activates) |
| `happy` | `Open Palm` | **`Positive Engagement`** |
| Any (e.g. `neutral`) | `Pointing` | **`Requesting Attention`** |
| *Any other combination* | *Any other combination* | **`Normal`** |

---

## Future Roadmap: Custom TFLite MobileNetV2 Emotion Model

To optimize inference speed and reduce the installation size (removing TensorFlow and MTCNN packages), we can replace the `fer` library with a lightweight **TensorFlow Lite (TFLite) MobileNetV2** model trained on FER-2013:

1. **Replace Dependencies**:
   Remove `fer` and `tensorflow` from `requirements.txt`. Add `tflite-runtime` (or `opencv-python`'s DNN module which supports running TFLite models directly).
2. **Model Setup**:
   Place a quantized model file `emotion_model_quant.tflite` (typically ~2-5 MB) in the project directory.
3. **Rewrite `emotion_model.py`**:
   Load the TFLite interpreter and run inference:
   ```python
   import cv2
   import numpy as np
   import tflite_runtime.interpreter as tflite

   interpreter = tflite.Interpreter(model_path="emotion_model_quant.tflite")
   interpreter.allocate_tensors()
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()

   def get_emotion(frame: np.ndarray) -> tuple[str, float]:
       # 1. Use OpenCV Haar Cascade to crop the face region (very fast)
       # 2. Resize the face crop to match the model input size (e.g., 224x224 or 48x48)
       # 3. Convert to grayscale or RGB and normalize pixels
       # 4. Set tensor, invoke, and extract the softmax probabilities
       # 5. Return (emotion_label, confidence)
       pass
   ```
4. **Benefits**:
   - Reduces project memory footprint by >90%.
   - Speeds up emotion frame processing from ~150ms to <15ms on standard CPUs, allowing emotion detection on every frame instead of every 5th frame.
