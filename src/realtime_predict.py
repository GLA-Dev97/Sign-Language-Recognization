import argparse
import json

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


IMG_SIZE = 64


def load_class_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    return {int(v): k for k, v in class_indices.items()}


def extract_hand_roi(frame, hand_landmarks, padding=20):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)
    x_max = min(max(xs) + padding, w)
    y_max = min(max(ys) + padding, h)

    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)


def preprocess(roi):
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi


def main():
    parser = argparse.ArgumentParser(description="Run real-time sign language prediction.")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    parser.add_argument("--class_map", default="models/class_indices.json", help="JSON class mapping path")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    idx_to_class = load_class_map(args.class_map)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            pred_label = "No hand detected"
            confidence_text = ""

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                roi, (x1, y1, x2, y2) = extract_hand_roi(frame, hand_landmarks)

                if roi.size > 0:
                    x_input = preprocess(roi)
                    probs = model.predict(x_input, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    conf = float(np.max(probs))

                    pred_label = idx_to_class.get(pred_idx, "Unknown")
                    confidence_text = f"Confidence: {conf * 100:.2f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"Prediction: {pred_label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
