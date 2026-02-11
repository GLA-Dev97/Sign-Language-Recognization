# Sign Language Recognition System (BCA Final Year Project)

This project helps you build a **complete Sign Language Recognition System** using Python. It is designed in simple language for BCA students and includes dataset collection, CNN training, and real-time webcam prediction.

---

## 1) Project Overview

### What is Sign Language Recognition?
Sign Language Recognition is a computer vision + AI system that reads hand gestures from camera/video and converts them into understandable text.

### Problem Statement
Many hearing-impaired people use sign language, but many others do not understand it. This creates a communication gap in classrooms, offices, hospitals, and public places.

### Objective of the Project
- Detect hand gestures using webcam.
- Train a deep learning model to classify gestures.
- Show recognized sign as text on screen in real time.
- Build an affordable, practical student-level prototype.

### Real-World Applications
- Communication aid for hearing/speech impaired users.
- Smart classrooms and inclusive education.
- Hospital and emergency desk communication support.
- Human-computer interaction and gesture-based control.

---

## 2) Technologies to Use

- **Python**: main programming language.
- **OpenCV**: webcam/video processing.
- **MediaPipe**: hand detection and tracking.
- **TensorFlow / Keras**: CNN model training and prediction.
- **NumPy, Pandas, Matplotlib**: data handling and plotting.
- **Scikit-learn**: confusion matrix, precision, recall, classification report.

---

## 3) System Architecture

1. **Webcam Input**
2. **Hand Landmark Detection (MediaPipe Hands)**
3. **Hand ROI Extraction (bounding box crop)**
4. **Dataset Creation (label-wise image storage)**
5. **Data Preprocessing (resize, normalize)**
6. **Model Training (CNN classifier)**
7. **Real-time Prediction**
8. **Output Display (predicted text on frame)**

```text
Webcam -> MediaPipe Hands -> Hand Crop ROI -> CNN Model -> Predicted Sign -> Display Text
```

---

## 4) Dataset Creation

### How to Collect Hand Gesture Images
Use `src/collect_dataset.py`:
- Select a label (example: `A`, `B`, `C`, `HELLO`, `THANKS`).
- Script opens webcam.
- MediaPipe detects hand.
- Hand region is cropped and saved when you press **S**.

### How Many Samples Per Gesture?
- Minimum: **300 images per gesture**.
- Recommended for better results: **700+ images per gesture**.

### Folder Structure

```text
Sign-Language-Recognization/
├── data/
│   ├── raw/
│   │   ├── A/
│   │   ├── B/
│   │   └── C/
│   └── processed/            # optional, if you create separate preprocessed data
├── models/
├── outputs/
└── src/
```

### Data Preprocessing
- Detect hand and crop ROI.
- Resize to `64 x 64`.
- Convert BGR to RGB.
- Normalize pixels to `0-1`.

### Train-Test Split
Use Keras `validation_split` in `ImageDataGenerator`:
- Training: **80%**
- Validation/Test: **20%**

---

## 5) Model Building (CNN)

### Input Shape
`(64, 64, 3)`

### Suggested CNN Layers
- Conv2D + ReLU + MaxPool
- Conv2D + ReLU + MaxPool
- Conv2D + ReLU + MaxPool
- Flatten
- Dense + ReLU + Dropout
- Output Dense + Softmax

### Activation Functions
- Hidden layers: `ReLU`
- Output: `Softmax`

### Compilation
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

### Training Process
- Use augmented batches via `ImageDataGenerator`.
- Train for `20-30 epochs`.
- Save best model using `ModelCheckpoint`.

### Save Model
Saved as `.h5` file, example:
`models/sign_language_cnn.h5`

---

## 6) Real-Time Prediction

The script:
1. Loads trained `.h5` model.
2. Reads webcam frames.
3. Uses MediaPipe to locate hand.
4. Crops and preprocesses hand region.
5. Predicts class using CNN.
6. Displays gesture text and confidence.

Run:
```bash
python src/realtime_predict.py --model models/sign_language_cnn.h5
```

---

## 7) Required Libraries Installation

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Direct install command:
```bash
pip install opencv-python mediapipe tensorflow numpy pandas matplotlib scikit-learn seaborn
```

---

## 8) Complete Python Code

All complete scripts are in `src/`:
- `collect_dataset.py` -> dataset collection
- `train_model.py` -> CNN training + evaluation
- `realtime_predict.py` -> real-time prediction

You can run in this order:
```bash
python src/collect_dataset.py --label A --num_samples 500
python src/collect_dataset.py --label B --num_samples 500
python src/collect_dataset.py --label C --num_samples 500

python src/train_model.py --data_dir data/raw --epochs 25

python src/realtime_predict.py --model models/sign_language_cnn.h5
```

---

## 9) Accuracy Improvement Tips

- **Data augmentation**: rotation, zoom, brightness change, flip.
- **Background control**: keep clean background while collecting data.
- **Lighting control**: consistent bright light, avoid dark/noisy frames.
- **Increase dataset size**: more samples from different users and angles.
- **Balanced classes**: same count for each gesture class.

---

## 10) Evaluation Metrics

From `train_model.py`:
- **Accuracy**: percentage of correct predictions.
- **Confusion Matrix**: class-wise correct/wrong predictions.
- **Precision**: out of predicted class samples, how many are correct.
- **Recall**: out of actual class samples, how many are detected.

Outputs saved in `outputs/`:
- `training_history.png`
- `confusion_matrix.png`
- `classification_report.csv`

---

## 11) Hardware & Software Requirements

### Minimum Hardware
- CPU: Intel i3/Ryzen 3 or higher
- RAM: 8 GB recommended (4 GB minimum)
- Webcam: built-in or USB
- Storage: 2-5 GB free

### Software
- OS: Windows/Linux/macOS
- Python: **3.10 or 3.11 recommended**
- IDE: VS Code / PyCharm / Jupyter

---

## 12) Future Enhancements

- Combine characters into **words/sentences**.
- Add **text-to-speech** voice output.
- Build **mobile app** using TensorFlow Lite.
- Deploy as **cloud API/web app** for remote usage.
- Add **two-hand and dynamic gesture** recognition.

---

## 13) Project Report Content (BCA Format)

Include these chapters:
1. **Abstract**
2. **Introduction**
3. **Literature Review**
4. **Problem Statement & Objectives**
5. **Methodology / System Design**
6. **Implementation**
7. **Results and Evaluation**
8. **Conclusion**
9. **Future Scope**
10. **References**

---

## Quick Demo Flow

1. Install dependencies.
2. Collect images for each sign.
3. Train model and check confusion matrix.
4. Run real-time prediction and show output text.
5. Use screenshots/metrics in final report.

Good luck with your BCA final year project! 🎓
