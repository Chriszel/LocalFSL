import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, BatchNormalization, Add, LSTM, TimeDistributed, GlobalAveragePooling3D, Reshape
from tensorflow.keras.utils import to_categorical

# ✅ Set Paths for Dataset
DATA_PATH = "D:/THESIS COMPILE/NEW FEB28/"
X_train_path = DATA_PATH + "train_data.npy"
y_train_path = DATA_PATH + "train_labels.npy"
X_test_path = DATA_PATH + "test_data.npy"
y_test_path = DATA_PATH + "test_labels.npy"

# ✅ Load Data
def load_data():
    try:
        X_train = np.load(X_train_path, mmap_mode='r').astype(np.float32)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path, mmap_mode='r').astype(np.float32)
        y_test = np.load(y_test_path)

        if y_train.size == 0 or y_test.size == 0:
            raise ValueError("❌ Loaded dataset is empty!")

        print(f"✅ Data Loaded! X_train: {X_train.shape}, X_test: {X_test.shape}")

        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        print(f"✅ One-hot encoding complete. Classes: {num_classes}")

        return X_train, y_train, X_test, y_test, num_classes
    except Exception as e:
        print("❌ Error loading data:", e)
        return None, None, None, None, None

# ✅ Compute Class Weights
def compute_weights(y_train):
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train_labels), y=y_train_labels)
    return dict(enumerate(class_weights))

# ✅ Build 3D CNN + LSTM Model
def build_3d_cnn_lstm(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN Feature Extraction
    x = Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    residual = Conv3D(128, kernel_size=(1, 1, 1), padding="same")(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # ✅ Reshape for LSTM (Flatten spatial dimensions, keep time dimension)
    x = Reshape((x.shape[1], -1))(x)  # Convert (batch, 32, features)

    # LSTM Layer
    x = LSTM(256, return_sequences=False, dropout=0.7, recurrent_dropout=0.5)(x)

    # Fully Connected Layers
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.6)(x)
    output = Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, output)

# ✅ Train Model with K-Fold Cross-Validation & Generate Results Table
def train_kfold(X_train, y_train, num_classes, n_splits=5, epochs=100, batch_size=8):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_weights = compute_weights(y_train)

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_3d_cnn_lstm(input_shape=X_train.shape[1:], num_classes=num_classes)
        optimizer = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)  # Gradient Clipping
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = callbacks.ModelCheckpoint(
            f"D:/THESIS COMPILE/NEW FEB28/best_model_fold_{fold_idx + 1}.keras",
            monitor='val_accuracy', save_best_only=True, verbose=1)

        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-6)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, class_weight=class_weights,
                  verbose=1, callbacks=[checkpoint, lr_scheduler, early_stop])

        model.save(f"D:/THESIS COMPILE/NEW FEB28/3D_CNN_LSTM_Fold_{fold_idx + 1}.keras")

        # ✅ Evaluate Model
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average="weighted")

        results.append([fold_idx + 1, accuracy, precision, recall, f1])

    # ✅ Convert results to DataFrame & Display Table
    df_results = pd.DataFrame(results, columns=["Fold", "Accuracy", "Precision", "Recall", "F1-Score"])
    df_results.loc["Average"] = df_results.mean()
    print("\n=== Cross-Validation Metrics Summary ===")
    print(df_results)

# ✅ Main Routine
def main():
    X_train, y_train, X_test, y_test, num_classes = load_data()
    if X_train is None:
        return
    train_kfold(X_train, y_train, num_classes, n_splits=5, epochs=100, batch_size=8)

if __name__ == "__main__":
    main()
