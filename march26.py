import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# === ✅ 1. PATHS ===
DATA_PATH = "D:/THESIS COMPILE/NEW FEB28/"
X_train_path = DATA_PATH + "train_data.npy"
y_train_path = DATA_PATH + "train_labels.npy"
X_test_path = DATA_PATH + "test_data.npy"
y_test_path = DATA_PATH + "test_labels.npy"

# === ✅ 2. LOAD & PREPROCESS ===
def load_data():
    try:
        X_train = np.load(X_train_path, mmap_mode='r').astype(np.float32)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path, mmap_mode='r').astype(np.float32)
        y_test = np.load(y_test_path)

        if y_train.size == 0 or y_test.size == 0:
            raise ValueError("❌ Loaded dataset is empty!")

        # Normalize video frames to 0–1 range
        X_train /= 255.0
        X_test /= 255.0

        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return X_train, y_train, X_test, y_test, num_classes
    except Exception as e:
        print("❌ Error loading data:", e)
        return None, None, None, None, None

# === ✅ 3. CLASS WEIGHTS ===
def compute_weights(y_train):
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train_labels), y=y_train_labels)
    return dict(enumerate(class_weights))

# === ✅ 4. MODEL: TimeDistributed CNN + LSTM ===
def build_tdcnn_lstm(input_shape, num_classes):
    inputs = Input(shape=input_shape)  # (timesteps, H, W, C)

    x = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Flatten())(x)
    x = LSTM(128, dropout=0.5, recurrent_dropout=0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, output)

# === ✅ 5. K-FOLD TRAINING ===
def train_kfold(X_train, y_train, num_classes, n_splits=5, epochs=100, batch_size=8):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_weights = compute_weights(y_train)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_tdcnn_lstm(input_shape=X_train.shape[1:], num_classes=num_classes)
        optimizer = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # ✅ Label smoothing
            metrics=["accuracy", TopKCategoricalAccuracy(k=5)]  # ✅ Top-5 accuracy
        )

        checkpoint = callbacks.ModelCheckpoint(
            f"{DATA_PATH}tdcnn_lstm_best_model_fold_{fold_idx + 1}.keras",
            monitor='val_accuracy', save_best_only=True, verbose=1)

        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-6)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, class_weight=class_weights,
                  verbose=1, callbacks=[checkpoint, lr_scheduler, early_stop])

        model.save(f"{DATA_PATH}TDCNN_LSTM_Fold_{fold_idx + 1}.keras")

        # ✅ Evaluation
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average="weighted")
        results.append([fold_idx + 1, accuracy, precision, recall, f1])

    df_results = pd.DataFrame(results, columns=["Fold", "Accuracy", "Precision", "Recall", "F1-Score"])
    df_results.loc["Average"] = df_results.mean()
    print("\n=== Cross-Validation Metrics Summary ===")
    print(df_results)
    df_results.to_csv(f"{DATA_PATH}tdcnn_lstm_cv_results.csv", index=False)

# === ✅ MAIN ===
def main():
    X_train, y_train, X_test, y_test, num_classes = load_data()
    if X_train is None:
        return
    train_kfold(X_train, y_train, num_classes, n_splits=5, epochs=100, batch_size=8)

if __name__ == "__main__":
    main()
