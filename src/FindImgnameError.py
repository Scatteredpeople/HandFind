import pandas as pd
import re

# 从log.txt中提取图像名称的函数，并去掉.jpg扩展名
def extract_image_names(log_path):
    image_names = []
    with open(log_path, 'r') as file:
        log_content = file.readlines()
    for line in log_content:
        match = re.search(r'No hands detected in image (\S+)', line)
        if match:
            image_name = match.group(1).strip()  # 去除可能存在的多余空格
            image_name_without_ext = image_name.replace('.jpg', '')  # 去掉.jpg扩展名
            image_names.append(image_name_without_ext)
            print(f"Extracted image name (without .jpg): {image_name_without_ext}")  # 调试信息
    return image_names

# 定义log文件路径和csv文件路径
log_path = r'.\data1\log.txt'
csv_path = r'.\data1\find_hand.csv'

# 提取需要删除的图片ID列表
images_to_remove = extract_image_names(log_path)

# 检查提取的图像名称列表
print(f"Images to remove: {images_to_remove}")

# 读取CSV文件，跳过损坏的行
try:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# 检查CSV文件的列名
if 'Image_ID' not in df.columns:
    print("CSV file does not contain an 'Image_ID' column.")
    exit()

# 打印CSV文件的前几行以验证列名和数据
print("First few rows of the CSV file:")
print(df.head())

# 验证提取的图像名称是否存在于CSV文件中
existing_images_in_csv = set(df['Image_ID'])
non_existent_images = [img for img in images_to_remove if img not in existing_images_in_csv]

if non_existent_images:
    print(f"The following images were not found in the CSV file: {non_existent_images}")
else:
    print("All extracted images are present in the CSV file.")

# 删除DataFrame中Image_ID在images_to_remove中的行
df_filtered = df[~df['Image_ID'].isin(images_to_remove)]

# 打印过滤前后的行数
print(f"Original DataFrame shape: {df.shape}")
print(f"Filtered DataFrame shape: {df_filtered.shape}")

# 将过滤后的数据保存回CSV文件
output_csv_path = r'.\data1\find.csv'
df_filtered.to_csv(output_csv_path, index=False)

print(f"Filtered CSV has been saved to {output_csv_path}")
