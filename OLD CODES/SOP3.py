import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --------------------------
# ✅ 1. Generate Small Dataset to Simulate Overfitting
# --------------------------
def generate_data(num_samples=100, num_classes=5):
    """Generates a small dataset that leads to overfitting."""
    X = np.random.rand(num_samples, 16, 64, 64, 3).astype(np.float32)  # 16-frame videos, 64x64 RGB
    y = np.random.randint(0, num_classes, num_samples)
    y = to_categorical(y, num_classes)
    return X, y

# Small dataset causes overfitting
X_train, y_train = generate_data(num_samples=100, num_classes=5)
X_test, y_test = generate_data(num_samples=20, num_classes=5)

# --------------------------
# ✅ 2. Define Overfitting-Prone 3D CNN Model (Fixed Pooling Issue)
# --------------------------
def build_fixed_overfitting_model(input_shape, num_classes):
    """Builds a deep 3D CNN model without regularization but prevents negative dimensions."""
    input_layer = Input(shape=input_shape)
    
    x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)  # Adjusted pooling to prevent depth shrinking too fast
    
    x = Conv3D(128, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)  # Keeps depth stable longer
    
    x = Conv3D(256, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)  # Now safely reduce all dimensions
    
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)  # Large dense layers prone to overfitting
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

# --------------------------
# ✅ 3. Train Model Without Regularization (To Simulate Overfitting)
# --------------------------
input_shape = (16, 64, 64, 3)
num_classes = 5

model = build_fixed_overfitting_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  # Longer training to show overfitting
    batch_size=8,
    verbose=1
)

# --------------------------
# ✅ 4. Visualizing Overfitting (Training vs. Validation Accuracy & Loss)
# --------------------------
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Overfitting Effect on Accuracy")
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Validation Loss", marker="s", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Overfitting Effect on Loss")
plt.legend()
plt.grid(True)

plt.show()