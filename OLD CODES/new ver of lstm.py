import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, TimeDistributed, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.utils import to_categorical

# --------------------------
# ✅ 1. Environment Setup
# --------------------------
print("✅ Initializing script...")

# ✅ Added Optimizations for Performance
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=false"
os.environ["TF_DETERMINISTIC_OPS"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ⬆ Reduce memory usage
tf.config.threading.set_inter_op_parallelism_threads(2)  # ⬆ Optimize CPU performance
tf.config.threading.set_intra_op_parallelism_threads(2)

print("✅ TensorFlow imported...")

# --------------------------
# ✅ 2. Data Loading & Preprocessing
# --------------------------
def load_data():
    """Loads dataset from .npy files, normalizes, and prepares for training."""
    try:
        X_train_path = "D:/THESIS COMPILE/NEW FEB28/train_data.npy"
        y_train_path = "D:/THESIS COMPILE/NEW FEB28/train_labels.npy"
        X_test_path = "D:/THESIS COMPILE/NEW FEB28/test_data.npy"
        y_test_path = "D:/THESIS COMPILE/NEW FEB28/test_labels.npy"

        if not (os.path.exists(X_train_path) and os.path.exists(y_train_path) and 
                os.path.exists(X_test_path) and os.path.exists(y_test_path)):
            raise FileNotFoundError("❌ One or more dataset files not found!")
#here float32
        X_train = np.load(X_train_path).astype('float32') / 255.0  # Normalize to [0,1]
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path).astype('float32') / 255.0  # Normalize to [0,1]
        y_test = np.load(y_test_path)

        if y_train.size == 0 or y_test.size == 0:
            raise ValueError("❌ Loaded dataset is empty!")

        print(f"✅ Data loaded successfully! X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # One-hot encode labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        print(f"✅ One-hot encoding complete. Number of classes: {num_classes}")

        return X_train, y_train, X_test, y_test, num_classes
    except Exception as e:
        print("❌ Error loading data:", e)
        return None, None, None, None, None

# --------------------------
# ✅ 3. K-Fold Data Generator HERE BATCH SIZE
# --------------------------
class VideoDataGenerator(tf.keras.utils.Sequence):
    """Efficiently loads video batches and resizes frames to 96x96."""
    def __init__(self, X, y, batch_size=16, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indexes]
        batch_y = self.y[batch_indexes]

        return np.array(batch_X), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --------------------------
# ✅ 4. Transfer Learning Model
# --------------------------
def build_transfer_learning_model(input_shape, num_classes):
    """Builds a transfer learning model using MobileNetV2 + Bidirectional LSTM."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    
    # Fine-tune last 50 layers
    for layer in base_model.layers[:-50]:  
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    frame_model = models.Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D())
    ])
#BIDIRECTIONAL HERE
    video_input = Input(shape=input_shape)
    x = frame_model(video_input)
    x = Bidirectional(LSTM(256, return_sequences=False))(x)  # Use Bidirectional LSTM
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=video_input, outputs=outputs)
    return model

# --------------------------
# ✅ 5. K-Fold Training HERE BATCH 
# --------------------------
def train_kfold(X_train, y_train, num_classes, n_splits=4, epochs=50, batch_size=16):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_gen = VideoDataGenerator(X_tr, y_tr, batch_size=batch_size, augment=True)
        val_gen = VideoDataGenerator(X_val, y_val, batch_size=batch_size, augment=False)

        model = build_transfer_learning_model(input_shape=X_train.shape[1:], num_classes=num_classes)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])  # Lower LR

        history = model.fit(
            train_gen, 
            epochs=epochs, 
            validation_data=val_gen,
            verbose=1,
            callbacks=[
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )

        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        fold_accuracies.append(val_acc)
        print(f"Fold {fold_idx + 1} validation accuracy: {val_acc:.4f}")

    return fold_accuracies

# --------------------------
# ✅ 6. Main Routine
# -------------------------- LEARN RATE HERE , BATCH SIZE HERE 
def main():
    X_train, y_train, X_test, y_test, num_classes = load_data()
    if X_train is None:
        return

    print("\n🔹 Running K-Fold Cross-Validation Training...")
    fold_accuracies = train_kfold(X_train, y_train, num_classes, n_splits=4, epochs=50, batch_size=16)
    print("\nFinal K-Fold Accuracy:", fold_accuracies)

    # ✅ Define and train the final model on the entire dataset
    print("\n🔹 Training final model on entire training set...")
    final_model = build_transfer_learning_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    final_model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    final_model.fit(VideoDataGenerator(X_train, y_train, batch_size=16, augment=True), epochs=50, verbose=1)

    # ✅ Evaluate model
    print("\n🔹 Evaluating model on test data...")
    test_gen = VideoDataGenerator(X_test, y_test, batch_size=16, augment=False, shuffle=False)
    test_acc = final_model.evaluate(test_gen, verbose=1)[1]
    print(f"✅ Final model test accuracy: {test_acc:.4f}")

    final_model.save('kfold_video_classification_model.h5')
    print("✅ Final model saved!")

if __name__ == "__main__":
    main()
