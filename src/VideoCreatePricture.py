import cv2
import os

#从视频中提取关键帧，主要解决 视频数据到图像数据的转换需求，是计算机视觉项目（如手势识别）中常见的预处理步骤

# 视频文件夹路径
video_folder = "./data/video"

# 输出帧的文件夹路径
output_folder = "./data/picture"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取视频文件夹中的所有视频文件
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# 每秒提取的帧数
fps_target = 30

# 用于防止命名重复的字典
name_counter = {}

# 遍历每个视频文件
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件 {video_file}")
        continue

    # 获取视频的原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 防止fps_target大于视频原始帧率，确保frame_interval不为0
    frame_interval = max(int(fps / fps_target), 1)

    frame_count = 0
    frame_num = 0

    # 获取视频的基本名称（不包括扩展名）
    video_name = os.path.splitext(video_file)[0]

    # 初始化计数器
    if video_name not in name_counter:
        name_counter[video_name] = 0

    # 遍历视频的每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 每隔一定帧数提取一张帧
        if frame_count % frame_interval == 0:
            name_counter[video_name] += 1
            # 生成帧文件名，格式为 video_name_001.jpg, video_name_002.jpg 等
            output_filename = os.path.join(output_folder, f"{video_name}_{name_counter[video_name]:03d}.jpg")
            cv2.imwrite(output_filename, frame)

    # 释放视频对象
    cap.release()

print("帧提取完成")
