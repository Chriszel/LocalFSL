import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Load CSVs
base_path = "D:/THESIS COMPILE/NEW FEB28/clips"
labels_df = pd.read_csv("D:/THESIS COMPILE/NEW FEB28/labels.csv")
train_df = pd.read_csv("D:/THESIS COMPILE/NEW FEB28/train.csv")
test_df = pd.read_csv("D:/THESIS COMPILE/NEW FEB28/test.csv")

# Constants
FRAME_COUNT = 32 
FRAME_HEIGHT, FRAME_WIDTH = 64, 64 
CLIPS_DIR = base_path

def apply_motion_blur(image, kernel_size=15, angle=0):
    """
    Apply synthetic motion blur to an image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    rotation_matrix = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def extract_frames(video_path, frame_count=FRAME_COUNT, 
                   frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, apply_blur=False):
    """
    Reads a video and extracts 'frame_count' frames at evenly spaced intervals.
    Applies synthetic motion blur if 'apply_blur' is True.
    """
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_path}")
            return np.array([])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                if apply_blur:
                    angle = np.random.uniform(-30, 30)  # Random angle for motion blur
                    frame = apply_motion_blur(frame, kernel_size=15, angle=angle)

                # Resize to (64, 64)
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Convert to float32 and normalize to [0,1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)

        cap.release()
        return np.array(frames)
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return np.array([])

def process_dataset(df, base_dir, apply_blur=False):
    """
    Iterates over a DataFrame of video info, extracts frames, 
    and builds arrays for data and labels. Applies synthetic motion blur if 'apply_blur' is True.
    """
    data = []
    labels = []
    missing_videos = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_path = row['vid_path'].split(' ')[0].replace('\\', '/')

        # Fix duplicate "clips/clips/" issue
        if vid_path.startswith("clips/") or vid_path.startswith("clips\\"):
            video_path = os.path.join(base_dir, vid_path[len("clips/"):]).replace("\\", "/")
        else:
            video_path = os.path.join(base_dir, vid_path).replace("\\", "/")

        # Debugging print statement to check constructed paths
        print(f"Checking video path: {video_path}")

        if not os.path.exists(video_path):
            print(f"Missing: {video_path}")
            missing_videos += 1
            continue

        frames = extract_frames(video_path, apply_blur=apply_blur)
        if frames.shape[0] == FRAME_COUNT:
            data.append(frames)
            labels.append(row['label'])

    print(f"Missing videos: {missing_videos}")
    return np.array(data), np.array(labels)

def validate_preprocessing(train_df, test_df, 
                           train_data, test_data, 
                           train_labels, test_labels):
    """
    Validates preprocessing results:
      1. Video counts
      2. Frame dimensions
      3. Value ranges
      4. Label consistency
    """
    print("\nValidation Report:")
    print("-----------------")
    print(f"Training videos in CSV: {len(train_df)}")
    print(f"Processed training videos: {len(train_data)}")
    print(f"Testing videos in CSV: {len(test_df)}")
    print(f"Processed testing videos: {len(test_data)}")
    
    print("\nFrame Dimensions:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    print("\nValue Ranges:")
    print(f"Training data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    print(f"Testing data range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    unique_train_labels = set(train_labels)
    unique_test_labels = set(test_labels)
    print("\nLabel Statistics:")
    print(f"Number of unique training labels: {len(unique_train_labels)}")
    print(f"Number of unique testing labels: {len(unique_test_labels)}")
    
    def visualize_sample(data, labels, index):
        sample = data[index]
        middle_frame = sample[FRAME_COUNT//2]
        
        print(f"\nSample video {index}:")
        print(f"Label: {labels[index]}")
        print(f"Frame shape: {middle_frame.shape}")
        
        print(f"Mean pixel value: {middle_frame.mean():.3f}")
        print(f"Std pixel value: {middle_frame.std():.3f}")
    
    print("\nSample Visualization:")
    print("Training sample:")
    visualize_sample(train_data, train_labels, 0)
    print("\nTesting sample:")
    visualize_sample(test_data, test_labels, 0)
    
    return {
        'train_complete': len(train_data) == len(train_df),
        'test_complete': len(test_data) == len(test_df),
        'correct_dimensions': (
            train_data.shape[1:] == (FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3) and
            test_data.shape[1:] == (FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3)
        ),
        'proper_scaling': (
            0 <= train_data.min() <= train_data.max() <= 1 and
            0 <= test_data.min() <= test_data.max() <= 1
        )
    }

if __name__ == "__main__":
    print("Processing training data with motion blur...")
    train_data, train_labels = process_dataset(train_df, CLIPS_DIR, apply_blur=True)

    print("Processing testing data without motion blur...")
    test_data, test_labels = process_dataset(test_df, CLIPS_DIR, apply_blur=False)

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    np.save('train_data.npy', train_data)
    np.save('train_labels.npy', train_labels_encoded)
    np.save('test_data.npy', test_data)
    np.save('test_labels.npy', test_labels_encoded)

    validation_results = validate_preprocessing(
        train_df, test_df,
        train_data, test_data,
        train_labels, test_labels
    )

    print("\nValidation Summary:")
    print("-----------------")
    for check, passed in validation_results.items():
        print(f"{check}: {'✓' if passed else '✗'}")
