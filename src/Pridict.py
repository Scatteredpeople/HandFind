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
import random  # 导入random库

# 主要用于从静态图像中识别手势，并评估模型性能

# ======================
# 初始化MediaPipe配置
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils


# ======================
# 空间注意力模型定义
# ======================

class StaticGestureAttnModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=9):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = num_heads

        # 坐标嵌入层
        self.coord_embed = nn.Linear(input_dim, self.embed_dim)

        # 空间注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 层级特征融合
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
# 初始化模型和预处理工具
# ======================
# 加载预处理工具
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移动到GPU
model = StaticGestureAttnModel(
    input_dim=3,  # (x, y, z)
    num_classes=len(label_encoder.classes_),
    num_heads=8
).to(device)

# 加载预训练权重并切换到评估模式
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# ======================
# 全局变量和配置
# ======================
previous_landmarks = None
movement_threshold = 0.5
unique_predictions = set()
previous_prediction = None
true_labels = []
predicted_labels = []


# ======================
# 核心处理函数
# ======================
def extract_true_label(file_name):
    """从文件名提取真实标签"""
    match = re.match(r'([a-zA-Z]+)_\d+', file_name)
    return match.group(1) if match else None


def preprocess_features(landmarks):
    """预处理关节点数据为模型输入格式"""
    features = np.zeros(42 * 3, dtype=np.float32)

    for hand_idx, hand_landmarks in enumerate(landmarks[:2]):
        start_idx = hand_idx * 21 * 3
        for j, lm in enumerate(hand_landmarks.landmark):
            features[start_idx + j * 3] = lm.x
            features[start_idx + j * 3 + 1] = lm.y
            features[start_idx + j * 3 + 2] = lm.z

    # 标准化并重塑形状
    scaled = scaler.transform(features.reshape(1, -1))
    return scaled.reshape(1, 42, 3)  # [batch, 42, 3]


def process_hand_gesture_from_images(image_folder, frames_per_second=60):
    """处理图像文件夹中的手势识别"""
    global previous_landmarks, previous_prediction

    # 获取所有图片文件并随机打乱顺序
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)  # 在这里随机打乱顺序

    os.makedirs("saved_images", exist_ok=True)

    for idx, image_file in enumerate(image_files):
        frame = cv2.imread(os.path.join(image_folder, image_file))
        if frame is None:
            continue

        # 手部检测和关键点提取
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            print(f"未检测到手部: {image_file}")
            continue

        try:
            # 预处理特征
            input_data = preprocess_features(results.multi_hand_landmarks)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)  # 将输入数据移动到GPU

            # 模型推理
            with torch.no_grad():
                preds, attn_weights = model(input_tensor)

            # 解码预测结果
            pred_label = label_encoder.inverse_transform([preds.argmax().item()])[0]
            true_label = extract_true_label(image_file)

            # 记录结果
            true_labels.append(true_label)
            predicted_labels.append(pred_label)

            # 更新预测显示逻辑
            if pred_label != previous_prediction:
                print(f"预测更新: {pred_label}")
                previous_prediction = pred_label

            # 可视化标注
            draw_annotations(frame, results, true_label, pred_label)
            save_result_image(frame, image_file, idx)

        except Exception as e:
            print(f"处理失败 {image_file}: {str(e)}")
            continue

    cv2.destroyAllWindows()


def draw_annotations(frame, results, true_label, pred_label):
    """绘制标注信息"""
    color = (0, 255, 0) if true_label == pred_label else (0, 0, 255)

    # 绘制关键点
    for landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        for lm in landmarks.landmark:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # 添加文字标注
    cv2.putText(frame, f'True: {true_label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Pred: {pred_label}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def save_result_image(frame, filename, idx):
    """保存带标注的结果图像"""
    if idx % 5 == 0:  # 每5帧保存一张
        output_path = os.path.join("saved_images", f"result_{filename}")
        cv2.imwrite(output_path, frame)


# ======================
# 评估指标可视化
# ======================
def visualize_metrics():
    """生成并可视化评估指标"""
    # 分类报告
    unique_labels = list(set(true_labels + predicted_labels))
    print("\n分类报告:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=unique_labels))

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title("Confusion Matrix")
    plt.savefig("saved_images/confusion_matrix.png")
    plt.close()

    # 各分类指标可视化
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
# 主程序
# ======================
if __name__ == "__main__":
    image_folder = r".\data\picture_augmented"

    # 处理图像数据
    process_hand_gesture_from_images(image_folder)

    # 生成可视化结果
    visualize_metrics()
    print("处理完成！结果保存在 saved_images 目录")