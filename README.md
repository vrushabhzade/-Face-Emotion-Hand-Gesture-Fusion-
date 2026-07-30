# Real-Time Face Emotion + Hand Gesture Fusion Desktop App

A real-time Python desktop application that fuses facial emotion recognition (via the `fer` library using MTCNN face detection) and hand gesture classification (via `mediapipe` hand tracking). The application performs temporal smoothing (rolling window of 10 frames) and applies decision rules to classify user states into "ALERT: Possible Distress", "Positive Engagement", "Requesting Attention", or "Normal".

This repository contains the full source code, dependencies installation configuration, and detailed documentation.

---

## 📂 Project Structure

- **[.gitignore](file:///.gitignore)**: Configured to exclude virtual environments (`v/`), IDE config files, and caches.
- **[face-gesture-fusion/](file:///face-gesture-fusion/)**: Core application directory.
  - **[main.py](file:///face-gesture-fusion/main.py)**: Desktop application entry point, camera capture loop, frame overlays, and GUI dashboard.
  - **[gesture_utils.py](file:///face-gesture-fusion/gesture_utils.py)**: MediaPipe Hands tracking wrapper, finger extension states extractor, and gesture classification dictionary.
  - **[emotion_model.py](file:///face-gesture-fusion/emotion_model.py)**: Facial emotion recognition class wrapper using `fer.fer.FER` with MTCNN face localization.
  - **[fusion_logic.py](file:///face-gesture-fusion/fusion_logic.py)**: Rolling window majority vote smoothing and security status decision rules.
  - **[requirements.txt](file:///face-gesture-fusion/requirements.txt)**: List of pinned compatible Python libraries.
  - **[README.md](file:///face-gesture-fusion/README.md)**: Technical detail document explaining logic and future MobileNetV2 replacement.

---

## ⚙️ Setup and Installation

### Prerequisites
- Python **3.11** installed on Windows.
- A connected USB/Integrated Webcam.

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vrushabhzade/-Face-Emotion-Hand-Gesture-Fusion-.git
   cd "-Face-Emotion-Hand-Gesture-Fusion-"
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv v
   ```

3. **Activate the Virtual Environment**:
   - **On Windows (PowerShell)**:
     ```powershell
     .\v\Scripts\Activate.ps1
     ```
   - **On Windows (Command Prompt)**:
     ```cmd
     .\v\Scripts\activate.bat
     ```

4. **Install Dependencies**:
   We lock and align the packages to ensure compatibility with Python 3.11:
   ```bash
   pip install -r face-gesture-fusion/requirements.txt
   ```
   *Note: Pip will automatically downgrade `setuptools<70.0.0` to preserve the `pkg_resources` library required by `fer`.*

---

## 🚀 How to Run

Activate the virtual environment and execute `main.py` inside the `face-gesture-fusion` folder:
```bash
python face-gesture-fusion/main.py
```
- A window titled **"Face Emotion & Hand Gesture Fusion"** will open.
- The dashboard in the top-left displays your smoothed emotion, gesture name, fusion state, and real-time FPS.
- Press **`q`** while focusing on the camera window to quit.

---

## 🧠 Core Logic Overview

### 1. Finger-State Bit-Vector Mappings
Gestures are classified using a 5-bit vector: `[thumb, index, middle, ring, pinky]` where `1 = extended`, `0 = curled`.
- **Index, Middle, Ring, Pinky**: Tip y-coordinate is compared to PIP joint y-coordinate (`tip.y < pip.y` means extended).
- **Thumb**: Evaluates horizontal movement adjusted for anatomical handedness (Left hand: `tip.x > ip.x`; Right hand: `tip.x < ip.x`).

#### Supported Gestures:
- `[0, 0, 0, 0, 0]` -> **Fist**
- `[1, 1, 1, 1, 1]` -> **Open Palm**
- `[0, 1, 1, 0, 0]` -> **Peace**
- `[1, 0, 0, 0, 0]` -> **Thumbs Up**
- `[0, 1, 0, 0, 0]` -> **Pointing**
- `[1, 0, 0, 0, 1]` -> **Call Me**

### 2. State Fusion Rules
We compute a majority vote over a rolling history of 10 frames for both states to avoid noise. The smoothed states are fused using these rules:

| Emotion (Smoothed) | Gesture (Smoothed) | Fused Action Label |
|:---|:---|:---|
| `fear` or `angry` or `sad` | `Fist` | **`ALERT: Possible Distress`** (Red blinker pulses) |
| `happy` | `Open Palm` | **`Positive Engagement`** |
| Any (e.g. `neutral`) | `Pointing` | **`Requesting Attention`** |
| *Any other combination* | *Any other combination* | **`Normal`** |

---

## 🛠️ Custom TFLite MobileNetV2 Emotion Model Roadmap
To reduce dependency sizes (omitting TensorFlow and PyTorch) and optimize CPU processing speeds (from 150ms down to 10ms), you can replace the FER library with a custom TensorFlow Lite model:
1. Replace `fer` and `tensorflow` in `requirements.txt` with `tflite-runtime`.
2. Crop the face region using a lightweight Haar Cascade (`cv2.CascadeClassifier`).
3. Feed the normalized face crop (`48x48` or `224x224`) to a trained TFLite MobileNetV2 model.
4. Extract softmax probabilities for classification.
