import re
import os

# 从log.txt中提取图像名称的函数
def extract_image_names(log_path):
    image_names = []
    with open(log_path, 'r') as file:
        log_content = file.readlines()
    for line in log_content:
        match = re.search(r'No hands detected in image (\S+)', line)
        if match:
            image_name = match.group(1).strip()  # 去除可能存在的多余空格
            image_names.append(image_name)  # 保留.jpg扩展名
            print(f"Extracted image name: {image_name}")  # 调试信息
    return image_names

# 定义文件路径
log_path = r'.\data1\log.txt'
pictures_folder = r'.\data1\picture_augmented'

# 提取需要删除的图片ID列表
images_to_remove = extract_image_names(log_path)

# 检查提取的图像名称列表
print(f"Images to remove: {images_to_remove}")

# 删除pictures文件夹中对应的图像文件
for image_name in images_to_remove:
    image_file_path = os.path.join(pictures_folder, image_name)
    try:
        os.remove(image_file_path)
        print(f"Deleted image file: {image_file_path}")
    except FileNotFoundError:
        print(f"Image file not found: {image_file_path}")
    except Exception as e:
        print(f"Error deleting image file {image_file_path}: {e}")



