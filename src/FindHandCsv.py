import os  # 导入os模块用于路径操作
import cv2
import mediapipe as mp
import csv

#手势识别系统的数据预处理核心模块，主要完成从手势图片中 提取手部关键点坐标 并生成结构化数据集，为后续机器学习模型训练提供输入数据。

# 初始化mediapipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

def process_hand_gesture_from_images(input_folder, output_file):

    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建CSV文件并写入表头
    header = ["Image_ID"]  # 修改第一列为图片编号
    # Hand 1 landmarks (63 values: 21 x, y, z per hand)
    for i in range(1, 22):  # 21 keypoints
        header.append(f"Hand_1_Landmark_{i}_x")
        header.append(f"Hand_1_Landmark_{i}_y")
        header.append(f"Hand_1_Landmark_{i}_z")

    # Hand 2 landmarks (63 values: 21 x, y, z per hand)
    for i in range(1, 22):  # 21 keypoints
        header.append(f"Hand_2_Landmark_{i}_x")
        header.append(f"Hand_2_Landmark_{i}_y")
        header.append(f"Hand_2_Landmark_{i}_z")

    # 添加 labels 列
    header.append("Label")  # 新增标签列

    # 创建CSV文件并写入表头
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_files.sort()  # 按照文件名排序，确保处理顺序正确

    # 处理每张图片
    for image_file in image_files:
        # 获取图片文件名作为标签
        image_name = os.path.splitext(image_file)[0]  # 获取图片文件名（去除扩展名）

        # 构造图片路径并读取图片
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error opening image file {image_file}")
            continue

        # BGR 转 RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 进行手部关键点检测
        results = hands.process(rgb_image)

        row = []
        # 使用图片编号作为Image_ID
        image_id = image_name
        row.append(image_id)

        # 提取手势信息
        gesture = "Detected"  # 这里可以根据需求进行手势分类

        # 检查是否检测到手部
        if not results.multi_hand_landmarks:
            print(f"No hands detected in image {image_file}")
            row += [0] * 126  # 填充两只手的坐标为零
            row.append(image_name)
            # 导出手势和关键点数据到CSV文件
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            continue

        # 遍历检测到的每只手
        for hand_num, landmarks in enumerate(results.multi_hand_landmarks, 1):
            # 提取手势信息和关键点数据
            landmark_coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

            # 将每只手的关节坐标拆分到不同列
            for coord in landmark_coords:
                row.append(coord[0])  # x
                row.append(coord[1])  # y
                row.append(coord[2])  # z

            # 绘制手部关键点和连接线
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        # 如果只有1只手，第二只手的坐标用零填充
        if len(results.multi_hand_landmarks) == 1:
            row += [0] * 63  # 填充第二只手的63个位置为零

        # 如果检测到第二只手，确保其坐标正确记录
        elif len(results.multi_hand_landmarks) == 2:
            pass  # 第二只手已经正确记录，保持原样

        # 在行尾增加图片标签
        row.append(image_name)

        # 导出手势和关键点数据到CSV文件
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # 可选：显示识别结果
        cv2.imshow("Hand Gesture Recognition", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder = r".\data1\picture_augmented"  # 输入图片文件夹路径
    output_file = r".\data1\find_hand.csv"  # 输出CSV文件路径
 # 输出CSV文件路径
    process_hand_gesture_from_images(input_folder, output_file)
