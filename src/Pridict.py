import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import random  # Importing random library

# Mainly used to recognize gestures from static images and evaluate model performance

# ======================
# Initialize MediaPipe configuration
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils


# ======================
# Spatial Attention Model Definition
# ======================

class StaticGestureAttnModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=9):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = num_heads

        # Coordinate embedding layer
        self.coord_embed = nn.Linear(input_dim, self.embed_dim)

        # Spatial attention mechanism
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Hierarchical feature fusion
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * 42, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        embedded = self.coord_embed(x)
        attn_out, attn_weights = self.attn(embedded, embedded, embedded)
        fused = embedded + attn_out
        flattened = fused.view(fused.size(0), -1)
        return self.fc(flattened), attn_weights


# ======================
# Initialize model and preprocessing tools
# ======================
# Load preprocessing tools
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move it to GPU
model = StaticGestureAttnModel(
    input_dim=3,  # (x, y, z)
    num_classes=len(label_encoder.classes_),
    num_heads=8
).to(device)

# Load pre-trained weights and switch to evaluation mode
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# ======================
# Global variables and configurations
# ======================
previous_landmarks = None
movement_threshold = 0.5
unique_predictions = set()
previous_prediction = None
true_labels = []
predicted_labels = []


# ======================
# Core Processing Functions
# ======================
def extract_true_label(file_name):
    """Extract true label from file name"""
    match = re.match(r'([a-zA-Z]+)_\d+', file_name)
    return match.group(1) if match else None


def preprocess_features(landmarks):
    """Preprocess landmark data into model input format"""
    features = np.zeros(42 * 3, dtype=np.float32)

    for hand_idx, hand_landmarks in enumerate(landmarks[:2]):
        start_idx = hand_idx * 21 * 3
        for j, lm in enumerate(hand_landmarks.landmark):
            features[start_idx + j * 3] = lm.x
            features[start_idx + j * 3 + 1] = lm.y
            features[start_idx + j * 3 + 2] = lm.z

    # Standardize and reshape
    scaled = scaler.transform(features.reshape(1, -1))
    return scaled.reshape(1, 42, 3)  # [batch, 42, 3]


def process_hand_gesture_from_images(image_folder, frames_per_second=60):
    """Process gesture recognition from images in a folder"""
    global previous_landmarks, previous_prediction

    # Get all image files and shuffle their order
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)  # Shuffle the order here

    os.makedirs("saved_images", exist_ok=True)

    for idx, image_file in enumerate(image_files):
        frame = cv2.imread(os.path.join(image_folder, image_file))
        if frame is None:
            continue

        # Hand detection and landmark extraction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            print(f"Hand not detected: {image_file}")
            continue

        try:
            # Preprocess features
            input_data = preprocess_features(results.multi_hand_landmarks)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)  # Move input data to GPU

            # Model inference
            with torch.no_grad():
                preds, attn_weights = model(input_tensor)

            # Decode the prediction result
            pred_label = label_encoder.inverse_transform([preds.argmax().item()])[0]
            true_label = extract_true_label(image_file)

            # Record results
            true_labels.append(true_label)
            predicted_labels.append(pred_label)

            # Update prediction display logic
            if pred_label != previous_prediction:
                print(f"Prediction updated: {pred_label}")
                previous_prediction = pred_label

            # Visualize annotations
            draw_annotations(frame, results, true_label, pred_label)
            save_result_image(frame, image_file, idx)

        except Exception as e:
            print(f"Failed to process {image_file}: {str(e)}")
            continue

    cv2.destroyAllWindows()


def draw_annotations(frame, results, true_label, pred_label):
    """Draw annotation information"""
    color = (0, 255, 0) if true_label == pred_label else (0, 0, 255)

    # Draw key points
    for landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        for lm in landmarks.landmark:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Add text annotations
    cv2.putText(frame, f'True: {true_label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Pred: {pred_label}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def save_result_image(frame, filename, idx):
    """Save result image with annotations"""
    if idx % 5 == 0:  # Save one image every 5 frames
        output_path = os.path.join("saved_images", f"result_{filename}")
        cv2.imwrite(output_path, frame)


# ======================
# Visualization of Evaluation Metrics
# ======================
def visualize_metrics():
    """Generate and visualize evaluation metrics"""
    # Classification report
    unique_labels = list(set(true_labels + predicted_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=unique_labels))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.savefig("saved_images/confusion_matrix.png")
    plt.close()

    # Visualize per-class metrics
    metrics = {
        'Accuracy': accuracy_score(true_labels, predicted_labels),
        'Precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'Recall': recall_score(true_labels, predicted_labels, average='weighted'),
        'F1': f1_score(true_labels, predicted_labels, average='weighted')
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(y=list(metrics.values()), hue=list(metrics.keys()), palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.title("Performance Metrics")
    plt.savefig("saved_images/metrics_summary.png")
    plt.close()


# ======================
# Main Program
# ======================
if __name__ == "__main__":
    image_folder = r".\data\picture_augmented"

    # Process image data
    process_hand_gesture_from_images(image_folder)

    # Generate visualization results
    visualize_metrics()
    print("Processing complete! Results saved in the saved_images directory")
