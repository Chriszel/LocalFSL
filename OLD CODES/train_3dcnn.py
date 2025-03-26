import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

# Paths to the processed data
train_data_path = "D:/THESIS COMPILE/NEW FEB28/train_data.npy"
train_labels_path = "D:/THESIS COMPILE/NEW FEB28/train_labels_one_hot.npy"
test_data_path = "D:/THESIS COMPILE/NEW FEB28/test_data.npy"
test_labels_path = "D:/THESIS COMPILE/NEW FEB28/test_labels_one_hot.npy"

# Load data
if not os.path.exists(train_labels_path):
    raise FileNotFoundError(f"Error: {train_labels_path} not found! Run the one-hot encoding script first.")

train_data = np.load(train_data_path).astype(np.float32)
train_labels = np.load(train_labels_path)
test_data = np.load(test_data_path).astype(np.float32)
test_labels = np.load(test_labels_path)

# Normalize the data
train_data /= 255.0
test_data /= 255.0

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42
)

# Compute class weights
train_labels_original = np.argmax(train_labels, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_original),
    y=train_labels_original
)
class_weights_dict = dict(enumerate(class_weights))

# Define convolutional block function
def conv_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding,
               kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Define the 3D CNN model
def build_3d_cnn(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = conv_block(input_layer, filters=32)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = conv_block(x, filters=64)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = conv_block(x, filters=128)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

# Build and compile the model
input_shape = (train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4])
num_classes = train_labels.shape[1]
model = build_3d_cnn(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(filepath='model_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=16,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test data
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Classification Report
labels = [str(i) for i in range(predictions.shape[1])]
print(classification_report(true_classes, predicted_classes, target_names=labels))

# Confusion Matrix Plot
def plot_confusion_matrix(actual, predicted, labels, ds_type):
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    sns.set(font_scale=1.2)
    ax.set_title(f'Confusion Matrix for {ds_type}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(labels, rotation=90)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(true_classes, predicted_classes, labels, ds_type="Test Data")
