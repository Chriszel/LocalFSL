#HYBRID CNN-LSTM SIGN LANGUAGE RECOGNITION MODEL
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import mediapipe as mp

class SignLanguageRecognizer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hand_landmark_detector = mp.solutions.hands.Hands()
    
    def extract_hand_landmarks(self, video_frames):
        """
        Extract hand landmarks using MediaPipe
        
        Args:
            video_frames (np.array): Input video frames
        
        Returns:
            np.array: Extracted hand landmarks
        """
        landmarks_sequence = []
        
        for frame in video_frames:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            results = self.hand_landmark_detector.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Extract landmark coordinates
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                landmarks_sequence.append(landmarks)
            else:
                # If no landmarks detected, use zero padding
                landmarks_sequence.append([(0, 0, 0)] * 21)
        
        return np.array(landmarks_sequence)
    
    def build_hybrid_model(self):
        """
        Build a hybrid CNN-LSTM model for sign language recognition
        """
        model = models.Sequential([
            # CNN layers for spatial feature extraction
            layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),
            
            # Additional CNN layer
            layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),
            
            # Reshape for LSTM
            layers.Reshape((-1, np.prod(self.input_shape[1:]))),
            
            # LSTM layers for temporal sequence processing
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            
            # Dropout for regularization
            layers.Dropout(0.5),
            
            # Dense layers for classification
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def augment_data(self, X_train, y_train):
        """
        Apply data augmentation techniques
        
        Args:
            X_train (np.array): Training video frames
            y_train (np.array): Training labels
        
        Returns:
            Augmented dataset
        """
        augmented_X = []
        augmented_y = []
        
        for video, label in zip(X_train, y_train):
            # Original video
            augmented_X.append(video)
            augmented_y.append(label)
            
            # Horizontal flip
            flipped_video = np.flip(video, axis=2)
            augmented_X.append(flipped_video)
            augmented_y.append(label)
            
            # Random crop
            cropped_video = video[:, 10:90, 10:90, :]
            augmented_X.append(cropped_video)
            augmented_y.append(label)
        
        return np.array(augmented_X), np.array(augmented_y)

# Example usage
def main():
    # Placeholder dimensions - adjust based on FSL-105 dataset
    input_shape = (100, 100, 100, 3)  # (frames, height, width, channels)
    num_classes = 105  # Number of sign language signs
    
    recognizer = SignLanguageRecognizer(input_shape, num_classes)
    
    # Load and preprocess FSL-105 dataset
    X_train, y_train, X_test, y_test = load_fsl_dataset()
    
    # Data augmentation
    X_train_aug, y_train_aug = recognizer.augment_data(X_train, y_train)
    
    # Build and train model
    model = recognizer.build_hybrid_model()
    model.fit(
        X_train_aug, y_train_aug, 
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )