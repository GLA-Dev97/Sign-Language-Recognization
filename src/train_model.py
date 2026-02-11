import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (64, 64)
BATCH_SIZE = 32
DATA_SPLIT_SEED = 42


def build_cnn(num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path: str):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train CNN for sign language recognition.")
    parser.add_argument("--data_dir", default="data/raw", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--model_path", default="models/sign_language_cnn.h5")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
    )

    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=DATA_SPLIT_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=DATA_SPLIT_SEED,
    )

    # Use a dedicated holdout iterator for metrics/reporting so evaluation is
    # deterministic and independent from the validation stream used during fit.
    eval_gen = eval_datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=DATA_SPLIT_SEED,
    )

    num_classes = len(train_gen.class_indices)
    model = build_cnn(num_classes)

    checkpoint = ModelCheckpoint(args.model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
    )

    model.save(args.model_path)

    with open("models/class_indices.json", "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    plot_history(history, os.path.join(args.outputs, "training_history.png"))

    eval_gen.reset()
    probs = model.predict(eval_gen)
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_gen.classes

    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(args.outputs, "confusion_matrix.png"))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(args.outputs, "classification_report.csv")
    report_df.to_csv(report_path, index=True)

    print("\nTraining complete!")
    print(f"Model saved: {args.model_path}")
    print(f"Class mapping saved: models/class_indices.json")
    print(f"Plots/reports saved in: {args.outputs}")


if __name__ == "__main__":
    main()
