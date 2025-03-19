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

#基于空间注意力机制的手势识别深度学习系统

# ======================
# 数据预处理
# ======================

csv_path = r".\data1\change.csv"

data = pd.read_csv(csv_path)

# 提取特征和标签（假设最后一列为标签）
labels = data.iloc[:, -1].values
features = data.iloc[:, 1:-1].values  # 去除首尾列

# 标签编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# SMOTE过采样处理类别不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, encoded_labels)

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

assert X_train.shape[0] == y_train.shape[0], "训练集数据和标签数量不一致！"

# ======================
# 数据形状调整（处理双手数据）
# ======================
# 每个样本包含双手的42个关键点（21*2），每个关键点3个坐标
X_train = X_train.reshape(-1, 42, 3)
X_val = X_val.reshape(-1, 42, 3)
X_test = X_test.reshape(-1, 42, 3)


# 重新检查过采样后的 X_train 和 y_train 的长度是否一致
assert X_train.shape[0] == y_train.shape[0], f"X_train 和 y_train 样本数量不一致！X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"

# 确保 y_train_tensor 为一维
y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# 确保数据维度一致
assert X_train_tensor.shape[0] == y_train_tensor.shape[0], f"X_train_tensor 和 y_train_tensor 的 batch_size 不一致！X_train_tensor: {X_train_tensor.shape[0]}, y_train_tensor: {y_train_tensor.shape[0]}"

# 继续处理验证集和测试集
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).view(-1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).view(-1)

# ======================
# 模型定义（空间注意力架构）
class StaticGestureAttnModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8):
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
            nn.Linear(self.embed_dim * 42, 256),  # 拼接所有关键点特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x形状: [batch_size, 42, 3]
        embedded = self.coord_embed(x)  # [batch_size, 42, 64]

        # 自注意力
        attn_out, attn_weights = self.attn(embedded, embedded, embedded)
        fused = embedded + attn_out  # 残差连接

        # 特征拼接
        batch_size = fused.shape[0]
        flattened = fused.view(batch_size, -1)  # 展平关键点维度

        return self.fc(flattened), attn_weights


# ======================
# 训练配置
# ======================
num_classes = len(np.unique(y_resampled))
model = StaticGestureAttnModel(
    input_dim=3,          # 每个关键点的坐标维度（x,y,z）
    num_classes=num_classes,
    num_heads=8
)


# optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
# 调整优化器参数q
optimizer = Adam(model.parameters(),
                lr=1e-5,
                weight_decay=1e-4)

# 增加学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ======================
# 训练循环
# ======================
train_losses = []
val_accuracies = []
best_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(100):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 验证阶段
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

    # 记录指标
    train_losses.append(epoch_loss / len(train_loader))
    val_accuracies.append(val_acc)

    # 早停机制
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
# 测试评估
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

# 合并结果
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



# 提取模型倒数第二层特征
model.eval()
feature_extractor = nn.Sequential(*list(model.children())[:-1])
features = []
with torch.no_grad():
    for inputs, _ in test_loader:
        feat, _ = model(inputs)
        features.append(feat)
features = torch.cat(features).numpy()

# t-SNE降维
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(12,10))
sns.scatterplot(x=features_2d[:,0], y=features_2d[:,1],
                hue=label_encoder.inverse_transform(test_labels),
                palette='tab20', s=50, alpha=0.8)
plt.title("t-SNE Visualization of Feature Space")
plt.savefig("tsne_features.png")


# 混淆矩阵
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
    """综合可视化评估函数（修复类型错误版本）"""
    # 确保标签类型一致性
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    try:
        # 1. 分类指标雷达图
        plot_classification_radar(true_labels, pred_labels)

        # 2. 错误样本分析
        analyze_misclassified_samples(test_loader, true_labels, pred_labels)

        # 3. 关键点重要性分析
        analyze_keypoint_importance(attn_weights)
    except Exception as e:
        print(f"可视化过程中发生错误: {str(e)}")


def plot_classification_radar(true_labels, pred_labels):
    """绘制分类指标雷达图（修复类型错误）"""
    from sklearn.metrics import precision_recall_fscore_support
    import matplotlib.pyplot as plt
    from math import pi

    # 确保标签为整数类型
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    # 获取类别标签名称
    class_names = label_encoder.classes_

    # 计算各类别指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=np.unique(true_labels), average=None
    )

    # 准备雷达图数据
    categories = class_names
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 添加三个指标的曲线
    for metric, values, color in zip(
            ['Precision', 'Recall', 'F1-Score'],
            [precision, recall, f1],
            ['b', 'g', 'r']
    ):
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, color=color, linewidth=1, linestyle='solid', label=metric)
        ax.fill(angles, values, color=color, alpha=0.1)

    # 添加标签
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Classification Metrics Radar Chart")
    plt.savefig("classification_radar.png")
    plt.close()


def analyze_misclassified_samples(test_loader, true_labels, pred_labels):
    """错误样本分析可视化（添加类型检查）"""
    # 转换为numpy数组并确保类型一致
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)

    # 获取错误样本索引
    wrong_indices = np.where(true_labels != pred_labels)[0]

    if len(wrong_indices) == 0:
        print("没有错误分类样本")
        return

    # 随机选择3个错误样本
    np.random.seed(42)
    selected = np.random.choice(wrong_indices, min(3, len(wrong_indices)), replace=False)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected):
        # 获取样本数据
        sample = test_loader.dataset[idx][0].numpy()

        # 绘制关键点三维分布
        ax = plt.subplot(1, 3, i + 1, projection='3d')
        plot_3d_keypoints(sample, ax)

        # 添加标注（使用原始标签）
        true_name = label_encoder.inverse_transform([true_labels[idx]])[0]
        pred_name = label_encoder.inverse_transform([pred_labels[idx]])[0]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}")
    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    plt.close()


def plot_3d_keypoints(keypoints, ax):
    """绘制三维关键点（保持与训练相同的处理逻辑）"""
    # 重塑为[42, 3]的格式
    keypoints = keypoints.reshape(-1, 3)

    # 双手颜色区分
    colors = ['r', 'b']
    for hand in range(2):
        start = hand * 21
        end = start + 21
        x = keypoints[start:end, 0]
        y = keypoints[start:end, 1]
        z = keypoints[start:end, 2]

        ax.scatter(x, y, z, c=colors[hand], s=20)
        ax.plot(x, y, z, color=colors[hand], alpha=0.5)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)

# 关键点重要性热力图
def analyze_keypoint_importance(attn_weights_collector):
    """关键点重要性热力图（添加空值检查）"""
    if not attn_weights_collector:
        print("没有可用的注意力权重数据")
        return

    try:
        # 聚合所有注意力权重
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
        print(f"生成注意力热力图时出错: {str(e)}")


# ======================
# 在测试评估后调用增强可视化（修复调用方式）
# ======================
# 修改后的调用代码：
enhanced_visual_evaluation(
    test_labels.numpy().astype(int),  # 确保为整数类型
    test_preds.numpy().astype(int),  # 确保为整数类型
    attn_weights_collector,
    test_loader
)
# 保存预处理工具
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# 确保这行代码在整个训练过程结束后执行


model.load_state_dict(torch.load("best_model.pth", weights_only=True))  # 需要PyTorch 1.13+


