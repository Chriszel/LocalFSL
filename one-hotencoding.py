import numpy as np
from tensorflow.keras.utils import to_categorical
import os

# Paths to data
train_data_path = "D:/THESIS COMPILE/NEW FEB28/train_data.npy"
train_labels_path = "D:/THESIS COMPILE/NEW FEB28/train_labels.npy"
test_data_path = "D:/THESIS COMPILE/NEW FEB28/test_data.npy"
test_labels_path = "D:/THESIS COMPILE/NEW FEB28/test_labels.npy"

one_hot_train_labels_path = "D:/THESIS COMPILE/NEW FEB28/train_labels_one_hot.npy"
one_hot_test_labels_path = "D:/THESIS COMPILE/NEW FEB28/test_labels_one_hot.npy"

# Load data
train_data = np.load(train_data_path)
train_labels = np.load(train_labels_path)
test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)

# Convert labels to one-hot encoding if needed
num_classes = len(np.unique(train_labels))  # Get number of unique classes
train_labels_one_hot = to_categorical(train_labels, num_classes=num_classes)
test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)

# Save the one-hot encoded labels
np.save(one_hot_train_labels_path, train_labels_one_hot)
np.save(one_hot_test_labels_path, test_labels_one_hot)

print("One-hot encoding completed and saved successfully!")
