import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --------------------------
# ✅ 1. Simulating Motion Blur Effect (For SOP 2: Noise & Artifacts)
# --------------------------
def apply_motion_blur(frame, kernel_size=5):
    """Applies horizontal motion blur to a single frame."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(frame, -1, kernel)

def generate_video_data(num_samples=500, num_classes=5, blur_ratio=0.5):
    """Generates synthetic video data and applies motion blur to a fraction of them."""
    X = np.random.rand(num_samples, 16, 64, 64, 3).astype(np.float32)  # 16-frame videos, 64x64 RGB
    y = np.random.randint(0, num_classes, num_samples)
    y = to_categorical(y, num_classes)

    num_blur = int(num_samples * blur_ratio)
    for i in range(num_blur):
        for j in range(16):  # Apply blur to all frames in a video sample
            X[i, j] = apply_motion_blur(X[i, j])

    return X, y

# Generate dataset with 50% blurred videos
X_train, y_train = generate_video_data(num_samples=500, num_classes=5, blur_ratio=0.5)
X_test, y_test = generate_video_data(num_samples=100, num_classes=5, blur_ratio=0.5)

# --------------------------
# ✅ 2. Define 3D CNN Model (For SOP 1: Vanishing & Exploding Gradients)
# --------------------------
def build_vanishing_exploding_model(input_shape, num_classes):
    """A deep 3D CNN model with improper weight initialization, leading to gradient issues."""
    input_layer = Input(shape=input_shape)
    
    x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu")(input_layer)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation="relu")(x)
    x = Conv3D(256, kernel_size=(3, 3, 3), activation="relu")(x)
    x = Conv3D(512, kernel_size=(3, 3, 3), activation="relu")(x)  # Deep layers prone to vanishing gradients
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(1024, kernel_size=(3, 3, 3), activation="relu")(x)  # Likely to cause exploding gradients
    x = Flatten()(x)
    
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

# --------------------------
# ✅ 3. Train Overfitting-Prone Model (For SOP 3: Overfitting Issue)
# --------------------------
def build_overfitting_model(input_shape, num_classes):
    """Builds a 3D CNN model without regularization to simulate overfitting."""
    input_layer = Input(shape=input_shape)
    
    x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu")(input_layer)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), activation="relu")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Conv3D(256, kernel_size=(3, 3, 3), activation="relu")(x)
    x = Flatten()(x)
    
    x = Dense(512, activation="relu")(x)  # Large dense layers prone to overfitting
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

# --------------------------
# ✅ 4. Train & Evaluate Models
# --------------------------
input_shape = (16, 64, 64, 3)
num_classes = 5

# Train Model with Gradient Issues
model_vanishing_exploding = build_vanishing_exploding_model(input_shape, num_classes)
model_vanishing_exploding.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history_vanishing_exploding = model_vanishing_exploding.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1
)

# Train Model Prone to Overfitting
model_overfitting = build_overfitting_model(input_shape, num_classes)
model_overfitting.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history_overfitting = model_overfitting.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1
)

# --------------------------
# ✅ 5. Visualizing the Issues
# --------------------------
def plot_results(history, title):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(history_vanishing_exploding, "Vanishing & Exploding Gradient Effect")
plot_results(history_overfitting, "Overfitting Effect on Accuracy")
