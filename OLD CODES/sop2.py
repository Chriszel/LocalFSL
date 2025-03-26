import os
import numpy as np
import cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ================================
# ✅ 1. Configuration and Paths
# ================================
VIDEO_DIR = "D:/THESIS COMPILE/NEW FEB28/clips"
FRAME_COUNT = 16  # Frames per video
FRAME_WIDTH = 64
FRAME_HEIGHT = 64
VALID_FORMATS = (".mp4", ".avi", ".mov", ".MOV")  # Supported video formats

# ================================
# ✅ 2. Load Videos from Directory
# ================================
def load_videos_from_directory(directory):
    """Loads all videos from the dataset directory, splits into train/test automatically."""
    X, y = [], []

    if not os.path.exists(directory):
        print(f"❌ ERROR: Directory {directory} does not exist!")
        return np.array([]), np.array([])

    class_labels = sorted(os.listdir(directory))  # Folders are class labels
    if len(class_labels) == 0:
        print(f"❌ ERROR: No class folders found in {directory}!")
        return np.array([]), np.array([])

    for label in class_labels:
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue

        print(f"🔹 Loading videos from {class_dir} (Class {label})")
        for file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, file)

            # ✅ Allow `.MOV`, `.MP4`, `.AVI`
            if not video_path.endswith(VALID_FORMATS):  
                print(f"⚠️ Skipping {file}, unsupported format.")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Failed to open {file}, skipping...")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < FRAME_COUNT:
                print(f"⚠️ Skipping {video_path}, too few frames.")
                continue

            frame_indices = np.linspace(0, total_frames - 1, FRAME_COUNT, dtype=int)
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = frame / 255.0  # Normalize
                frames.append(frame)

            if len(frames) == FRAME_COUNT:
                X.append(frames)
                y.append(int(label))

            cap.release()

    if len(X) == 0:
        print("❌ ERROR: No videos successfully loaded!")

    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, len(class_labels))

    print(f"✅ Loaded {len(X)} videos from {directory} with {len(class_labels)} classes.")
    return X, y

# ✅ Load dataset
X, y = load_videos_from_directory(VIDEO_DIR)

# ✅ Split into train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y.argmax(axis=1), random_state=42)

# ================================
# ✅ 3. Apply Motion Blur to Dataset
# ================================
def apply_motion_blur(frame, kernel_size=5):
    """Applies horizontal motion blur to a single frame."""
    if kernel_size < 1:
        return frame

    kernel = np.zeros((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(frame, -1, kernel)

def apply_blur_to_dataset(X, blur_level=5):
    """Applies motion blur to the dataset."""
    if blur_level < 1:
        return X

    X_blurred = X.copy()
    for i in range(len(X_blurred)):
        for j in range(X_blurred.shape[1]):  # Iterate over frames
            X_blurred[i, j] = apply_motion_blur(X_blurred[i, j], kernel_size=max(1, blur_level))
    return X_blurred

# ================================
# ✅ 4. Define 3D CNN Model
# ================================
def build_3d_cnn(input_shape, num_classes):
    """Builds a 3D CNN model for video classification."""
    input_layer = Input(shape=input_shape)

    x = Conv3D(32, kernel_size=(3, 3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=input_layer, outputs=output_layer)

# ================================
# ✅ 5. Train & Evaluate Model
# ================================
input_shape = X_train.shape[1:]  
num_classes = y_train.shape[1]  
blur_levels = [0, 5, 10, 15, 20]  
results = []

for blur in blur_levels:
    print(f"\nTraining & Evaluating for Blur Level: {blur}")

    X_test_blurred = apply_blur_to_dataset(X_test, blur_level=blur)

    model = build_3d_cnn(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test_blurred, y_test),
        epochs=20,
        batch_size=16,
        verbose=1
    )

    y_pred = model.predict(X_test_blurred)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

    precision = report["weighted avg"]["precision"] * 100
    recall = report["weighted avg"]["recall"] * 100
    f1_score = report["weighted avg"]["f1-score"] * 100
    accuracy = report["accuracy"] * 100

    results.append((blur, accuracy, precision, recall, f1_score))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (Motion Blur Impact: {blur})")
    plt.show()

# ================================
# ✅ 6. Print Final Results
# ================================
print("\n--- 3D CNN Motion Blur Performance Analysis ---")
print(f"{'Blur Level':<10} | {'Accuracy':<8} | {'Precision':<8} | {'Recall':<8} | {'F1 Score':<8}")
print("-" * 55)

for blur, acc, prec, rec, f1 in results:
    print(f"{blur:<10} | {acc:.2f}%   | {prec:.2f}%   | {rec:.2f}%   | {f1:.2f}%   ")
