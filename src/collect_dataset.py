import argparse
import os
import time

import cv2
import mediapipe as mp


IMG_SIZE = 64


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def main():
    parser = argparse.ArgumentParser(description="Collect sign language gesture images.")
    parser.add_argument("--label", required=True, help="Gesture label name, e.g., A or HELLO")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of images to collect")
    parser.add_argument("--output_dir", default="data/raw", help="Root dataset folder")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    label_dir = os.path.join(args.output_dir, args.label)
    ensure_dir(label_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    saved_count = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
    last_save_time = 0

    print("\nControls:")
    print("  Press S to save sample")
    print("  Press Q to quit")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened() and saved_count < args.num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_roi = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_roi, (x1, y1, x2, y2) = extract_hand_roi(frame, hand_landmarks)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"Label: {args.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"Saved: {saved_count}/{args.num_samples}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, "Press S to save | Q to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Dataset Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s") and hand_roi is not None:
                now = time.time()
                if now - last_save_time < 0.1:
                    continue
                last_save_time = now

                hand_roi = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
                img_path = os.path.join(label_dir, f"{args.label}_{saved_count:05d}.jpg")
                cv2.imwrite(img_path, hand_roi)
                saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {saved_count} samples for label '{args.label}'.")


if __name__ == "__main__":
    main()
