# GestureSync AI 🎵🖐️

## Real-Time Gesture-Controlled Virtual Music Studio

GestureSync AI is a real-time computer vision and deep learning project that allows users to control a virtual music studio using hand gestures captured through a webcam.

The system detects hand landmarks using **MediaPipe**, classifies gestures using a custom **PyTorch deep learning model**, and converts recognized gestures into musical actions using **Pygame**.

The project combines **computer vision, deep learning, audio processing, and real-time human-computer interaction** into a single application.

---

## 🎯 Project Objectives

- Detect hand gestures in real time using a webcam.
- Extract hand landmark features using MediaPipe.
- Train a custom deep learning model for gesture classification.
- Control musical notes using hand gestures.
- Control virtual drums using hand position.
- Provide real-time gesture and confidence feedback.
- Build an interactive AI-based music application without requiring a dedicated GPU.

---

## ✨ Features

- 🖐️ Real-time hand tracking
- 🧠 Custom PyTorch gesture classification model
- 🎹 Virtual piano
- 🥁 Virtual drum machine
- 🎥 Webcam-based interaction
- 📊 Gesture confidence display
- 🔊 Real-time audio playback
- ⚡ Lightweight model suitable for CPU-based systems
- 📈 Model evaluation using accuracy, precision, recall, F1-score and confusion matrix

---

## 🎮 Gesture Controls

| Gesture | Action |
|---|---|
| 🖐️ Open Palm | Play C |
| 🤏 Pinch | Play selected piano note + drum |
| 👍 Thumbs Up | Play G |
| 👎 Thumbs Down | Play A |
| ✊ Fist | Stop audio |

### Pinch Interaction

The pinch gesture provides additional spatial control.

The horizontal position of the hand determines:

**Piano**

```text
C | D | E | F | G | A | B
```

**Drums**

```text
KICK | SNARE | HI-HAT
```

So, when the user performs a pinch, the system uses the hand's x-position to determine both the piano note and drum sound.

---

## 🧠 AI / Deep Learning Pipeline

The project follows this pipeline:

```text
Webcam
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Feature Extraction
   ↓
63 Numerical Features
   ↓
PyTorch Neural Network
   ↓
Gesture Classification
   ↓
Gesture Controller
   ↓
Piano / Drum Audio
```

---

## 🔍 Hand Landmark Features

MediaPipe detects **21 landmarks** on the hand.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

The coordinates are normalized relative to the wrist to make the model less dependent on the hand's position in the camera frame.

---

## 🤖 Deep Learning Model

A custom Multi-Layer Perceptron (MLP) was developed using PyTorch.

### Architecture

```text
Input: 63 features
        ↓
Linear: 128 neurons
        ↓
ReLU
        ↓
Dropout (0.2)
        ↓
Linear: 64 neurons
        ↓
ReLU
        ↓
Dropout (0.2)
        ↓
Linear: 32 neurons
        ↓
ReLU
        ↓
Output: 5 gesture classes
```

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Loss Function | Cross Entropy |
| Hidden Layers | 128 → 64 → 32 |
| Dropout | 0.2 |
| Input Features | 63 |
| Output Classes | 5 |

---

## 📊 Dataset

The dataset was created using webcam-based hand landmark collection.

### Gesture Classes

- Palm
- Pinch
- Fist
- Thumbs Up
- Thumbs Down

### Dataset Size

**751 samples**

| Gesture | Samples |
|---|---:|
| Palm | 309 |
| Thumbs Down | 145 |
| Fist | 113 |
| Thumbs Up | 102 |
| Pinch | 82 |
| **Total** | **751** |

The dataset was divided using a stratified split into:

- 70% Training
- 15% Validation
- 15% Testing

---

## 📈 Model Performance

The trained model achieved:

- **Validation Accuracy:** ~96.5%
- **Test Accuracy:** ~94.7%
- **Macro F1 Score:** ~0.94
- **Weighted F1 Score:** ~0.95

### Test Confusion Matrix

```text
[[16  1  0  0  0]
 [ 1 44  0  0  1]
 [ 1  0 11  0  0]
 [ 0  0  0 22  0]
 [ 1  0  1  0 14]]
```

The results demonstrate that the model can reliably distinguish between the five gesture classes under the collected testing conditions.

---

## 🛠️ Technologies Used

### AI / Computer Vision

- Python
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Scikit-learn

### Audio / Application

- Pygame
- WAV audio generation
- Real-time webcam processing

### Development

- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
GestureMusicStudio/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── hand_landmarker.task
│   ├── gesture_model.pth
│   └── label_mapping.json
│
├── audio/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── hand_tracker.py
│   ├── feature_extractor.py
│   ├── gesture_model.py
│   ├── predictor.py
│   ├── gesture_controller.py
│   ├── music_engine.py
│   ├── virtual_piano.py
│   └── drum_machine.py
│
├── training/
│   ├── collect_data.py
│   ├── train.py
│   └── evaluate.py
│
└── tests/
    └── test_music.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sahithy1024/GestureSync-AI.git
cd GestureSync-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Navigate to the `src` directory:

```powershell
cd src
python main.py
```

Make sure your webcam is connected.

The application will open a window showing:

- Webcam feed
- Detected hand landmarks
- Current gesture
- Gesture confidence
- Piano keyboard
- Drum zones
- Selected note
- Selected drum

Press:

```text
Q
```

to exit the application.

---

## 🔊 Audio System

The project uses Pygame for audio playback.

The virtual piano generates musical notes:

```text
C D E F G A B
```

The drum machine provides:

```text
Kick
Snare
Hi-Hat
```

Audio files are generated as WAV files and played through the Pygame mixer.

---

## 🧪 Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The project can also be extended to compare the neural network with traditional machine-learning algorithms such as:

- Random Forest
- Support Vector Machine (SVM)

---

## 🚧 Current Limitations

- Gesture recognition depends on lighting and camera quality.
- The model was trained on a relatively small custom dataset.
- Recognition may become less reliable with unusual hand orientations.
- Only one hand is currently used for gesture control.
- The current version provides basic musical interaction rather than full music composition.
- Gesture cooldown is used to prevent repeated accidental triggers.

---

## 🚀 Future Improvements

Possible future extensions include:

- 🎼 More musical notes and scales
- 🎹 Multiple octaves
- 🥁 More drum sounds
- 🎚️ Volume and tempo gesture controls
- ↔️ Swipe gestures
- 🎤 Voice + gesture hybrid control
- 🎧 MIDI controller integration
- 🎵 Recording and playback of generated melodies
- 👥 Multi-hand interaction
- 📱 Web-based interface
- 🤖 Larger and more diverse training dataset
- ⚡ Model optimization for lower latency

---

## 👥 Team Contribution

This project was developed as a collaborative deep-learning project.

### AI / Machine Learning

- Hand landmark processing
- Dataset collection
- Feature engineering
- PyTorch model development
- Model training
- Model evaluation
- Gesture prediction

### Application / Music System

- Pygame audio engine
- Virtual piano
- Drum machine
- Gesture-to-music interaction
- Real-time UI
- Application integration

### Joint Work

- System integration
- Testing
- Debugging
- Documentation
- Presentation

---

## 💡 Why This Project?

GestureSync AI demonstrates how deep learning can be integrated into a practical real-time application.

Instead of simply training a model and reporting its accuracy, the project connects the entire AI pipeline:

```text
Data Collection
      ↓
Feature Engineering
      ↓
Deep Learning
      ↓
Model Evaluation
      ↓
Real-Time Prediction
      ↓
Application Control
      ↓
Audio Output
```

This makes the project a practical example of deploying a trained machine-learning model into an interactive application.

---

## 📌 Project Status

**Status: Completed Prototype ✅**

The current version successfully demonstrates:

- Real-time hand tracking
- Custom gesture classification
- Deep learning inference
- Virtual piano interaction
- Virtual drum interaction
- Real-time audio output
- Gesture-based music control

---

## 📄 Project Documentation

Additional project documentation can include:

- Project presentation
- Case study/report
- Model evaluation results
- Dataset and training information

---

## ⭐ Highlights

- Custom PyTorch deep learning model
- 63 hand-landmark features
- 5 gesture classes
- 751 collected samples
- ~94.7% test accuracy
- Real-time webcam inference
- Virtual piano + drum machine
- CPU-friendly implementation
- End-to-end AI application

---

## 📜 License

This project is intended for educational and academic purposes.