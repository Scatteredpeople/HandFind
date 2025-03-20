import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Spatial Attention-Based Gesture Recognition Deep Learning System

# ======================
# Data Preprocessing
# ======================

csv_path = r".\data1\change.csv"

data = pd.read_csv(csv_path)

# Extract features and labels (assume the last column is the label)
labels = data.iloc[:, -1].values
features = data.iloc[:, 1:-1].values  # Remove the first and last columns

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# SMOTE oversampling to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, encoded_labels)

# Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

assert X_train.shape[0] == y_train.shape[0], "The number of training samples and labels are inconsistent!"

# ======================
# Reshape Data (Handling Two-Hand Data)
# ======================
# Each sample contains 42 keypoints for both hands (21*2), with each keypoint having 3 coordinates
X_train = X_train.reshape(-1, 42, 3)
X_val = X_val.reshape(-1, 42, 3)
X_test = X_test.reshape(-1, 42, 3)


# Recheck if the lengths of X_train and y_train are consistent after oversampling
assert X_train.shape[0] == y_train.shape[0], f"X_train and y_train sample numbers are inconsistent! X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"

# Ensure y_train_tensor is one-dimensional
y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Ensure consistent dimensions
assert X_train_tensor.shape[0] == y_train_tensor.shape[0], f"X_train_tensor and y_train_tensor batch sizes are inconsistent! X_train_tensor: {X_train_tensor.shape[0]}, y_train_tensor: {y_train_tensor.shape[0]}"

# Continue processing validation and test sets
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).view(-1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).view(-1)

# ======================
# Model Definition (Spatial Attention Architecture)
class StaticGestureAttnModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = num_heads

        # Coordinate Embedding Layer
        self.coord_embed = nn.Linear(input_dim, self.embed_dim)

        # Spatial Attention Mechanism
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Hierarchical Feature Fusion
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * 42, 256),  # Concatenate all keypoint features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, 42, 3]
        embedded = self.coord_embed(x)  # [batch_size, 42, 64]

        # Self-Attention
        attn_out, attn_weights = self.attn(embedded, embedded, embedded)
        fused = embedded + attn_out  # Residual connection

        # Feature Concatenation
        batch_size = fused.shape[0]
        flattened = fused.view(batch_size, -1)  # Flatten keypoint dimension

        return self.fc(flattened), attn_weights


# ======================
# Training Configuration
# ======================
num_classes = len(np.unique(y_resampled))
model = StaticGestureAttnModel(
    input_dim=3,          # Each keypoint's coordinate dimension (x,y,z)
    num_classes=num_classes,
    num_heads=8
)


# optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
# Adjust optimizer parameters q
optimizer = Adam(model.parameters(),
                lr=1e-5,
                weight_decay=1e-4)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ======================
# Training Loop
# ======================
train_losses = []
val_accuracies = []
best_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(100):
    # Training phase
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation phase
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs, _ = model(inputs)
            val_preds.append(outputs.argmax(dim=1))
            val_labels.append(labels)

    val_acc = accuracy_score(
        torch.cat(val_labels).numpy(),
        torch.cat(val_preds).numpy()
    )

    # Record metrics
    train_losses.append(epoch_loss / len(train_loader))
    val_accuracies.append(val_acc)

    # Early stopping mechanism
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch + 1:03d} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Val Acc: {val_acc:.4f}")

# ======================
# Test Evaluation
# ======================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_preds, test_labels = [], []
attn_weights_collector = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs, attn_weights = model(inputs)
        test_preds.append(outputs.argmax(dim=1))
        test_labels.append(labels)
        attn_weights_collector.append(attn_weights)

# Merge results
test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_acc = accuracy_score(test_labels.numpy(), test_preds.numpy())

print("\n" + "=" * 50)
print(f"Test Accuracy: {test_acc:.6f}")
print(classification_report(
    test_labels.numpy(),
    test_preds.numpy(),
    target_names=label_encoder.classes_,
    digits=6
))



# Extract second-to-last layer features
model.eval()
feature_extractor = nn.Sequential(*list(model.children())[:-1])
features = []
with torch.no_grad():
    for inputs, _ in test_loader:
        feat, _ = model(inputs)
        features.append(feat)
features = torch.cat(features).numpy()

# t-SNE dimensionality reduction
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Visualization
plt.figure(figsize=(12,10))
sns.scatterplot(x=features_2d[:,0], y=features_2d[:,1],
                hue=label_encoder.inverse_transform(test_labels),
                palette='tab20', s=50, alpha=0.8)
plt.title("t-SNE Visualization of Feature Space")
plt.savefig("tsne_features.png")


# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(15,12))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")


def enhanced_visual_evaluation(true_labels, pred_labels, attn_weights, test_loader):
    """Comprehensive visualization evaluation function (fixed type error version)"""
    # Ensure label type consistency
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    try:
        # 1. Classification Metrics Radar Chart
        plot_classification_radar(true_labels, pred_labels)

        # 2. Misclassified Samples Analysis
        analyze_misclassified_samples(test_loader, true_labels, pred_labels)

        # 3. Keypoint Importance Analysis
        analyze_keypoint_importance(attn_weights)
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")

def plot_classification_radar(true_labels, pred_labels):
    """Plot classification metrics radar chart (fix type error)"""
    from sklearn.metrics import precision_recall_fscore_support
    import matplotlib.pyplot as plt
    from math import pi

    # Ensure labels are integers
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    # Get class label names
    class_names = label_encoder.classes_

    # Calculate metrics for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=np.unique(true_labels), average=None
    )

    # Prepare data for the radar chart
    categories = class_names
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plot three metrics curves
    for metric, values, color in zip(
            ['Precision', 'Recall', 'F1-Score'],
            [precision, recall, f1],
            ['b', 'g', 'r']
    ):
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, color=color, linewidth=1, linestyle='solid', label=metric)
        ax.fill(angles, values, color=color, alpha=0.1)

    # Add labels
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Classification Metrics Radar Chart")
    plt.savefig("classification_radar.png")
    plt.close()


def analyze_misclassified_samples(test_loader, true_labels, pred_labels):
    """Misclassified sample analysis visualization (add type checking)"""
    # Convert to numpy arrays and ensure type consistency
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    # Get indices of misclassified samples
    wrong_indices = np.where(true_labels != pred_labels)[0]

    if len(wrong_indices) == 0:
        print("No misclassified samples")
        return

    # Randomly select 3 misclassified samples
    np.random.seed(42)
    selected = np.random.choice(wrong_indices, min(3, len(wrong_indices)), replace=False)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected):
        # Get sample data
        sample = test_loader.dataset[idx][0].numpy()

        # Plot 3D keypoint distribution
        ax = plt.subplot(1, 3, i + 1, projection='3d')
        plot_3d_keypoints(sample, ax)

        # Add labels (use original labels)
        true_name = label_encoder.inverse_transform([true_labels[idx]])[0]
        pred_name = label_encoder.inverse_transform([pred_labels[idx]])[0]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}")
    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    plt.close()


def plot_3d_keypoints(keypoints, ax):
    """Plot 3D keypoints (maintain the same logic as in training)"""
    # Reshape to [42, 3] format
    keypoints = keypoints.reshape(-1, 3)

    # Differentiate hands by color
    colors = ['r', 'b']
    for hand in range(2):
        start = hand * 21
        end = start + 21
        x = keypoints[start:end, 0]
        y = keypoints[start:end, 1]
        z = keypoints[start:end, 2]

        ax.scatter(x, y, z, c=colors[hand], s=20)
        ax.plot(x, y, z, color=colors[hand], alpha=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)

# Keypoint importance heatmap
def analyze_keypoint_importance(attn_weights_collector):
    """Keypoint importance heatmap (add null value checking)"""
    if not attn_weights_collector:
        print("No available attention weights data")
        return

    try:
        # Aggregate all attention weights
        all_attn = torch.stack([aw.mean(dim=0) for aw in attn_weights_collector]).mean(dim=0).cpu().numpy()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            all_attn,
            cmap="viridis",
            xticklabels=[f"Hand{1 + i // 21}-{i % 21}" for i in range(42)],
            yticklabels=[f"Hand{1 + i // 21}-{i % 21}" for i in range(42)],
            linewidths=0.5,
            annot=False
        )
        plt.title("Aggregated Attention Weights")
        plt.xlabel("Target Keypoint")
        plt.ylabel("Source Keypoint")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.savefig("aggregated_attention.png")
        plt.close()
    except Exception as e:
        print(f"Error generating attention heatmap: {str(e)}")


# ======================
# Call enhanced visualization after test evaluation (fix the call method)
# ======================
# Modified call code:
enhanced_visual_evaluation(
    test_labels.numpy().astype(int),  # Ensure integer type
    test_preds.numpy().astype(int),  # Ensure integer type
    attn_weights_collector,
    test_loader
)
# Save preprocessing tools
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Ensure this line is executed after the entire training process
model.load_state_dict(torch.load("best_model.pth", weights_only=True))  # Requires PyTorch 1.13+
