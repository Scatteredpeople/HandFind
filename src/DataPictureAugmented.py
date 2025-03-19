import os
import cv2
import numpy as np
import random
import math


def augment_images(input_dir, output_dir, num_augmentations=5):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有图片文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取原始图片
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 保存原始图片（可选）
            base_name, ext = os.path.splitext(filename)
            # cv2.imwrite(os.path.join(output_dir, f"{base_name}_original{ext}"), img)

            # 生成增强后的图片
            for i in range(num_augmentations):
                aug_img = img.copy()

                # 随机选择增强类型
                augmentation_type = random.choice(['rotate', 'flip', 'translate', 'noise'])

                if augmentation_type == 'rotate':
                    aug_img = rotate_image(aug_img)
                elif augmentation_type == 'flip':
                    aug_img = flip_image(aug_img)
                elif augmentation_type == 'translate':
                    aug_img = translate_image(aug_img)
                elif augmentation_type == 'noise':
                    aug_img = add_noise_to_image(aug_img)

                # 保存增强后的图片
                output_path = os.path.join(
                    output_dir,
                    f"{base_name}_aug{i}_{augmentation_type}{ext}"
                )
                cv2.imwrite(output_path, aug_img)


def rotate_image(img, max_angle=45):
    angle = random.uniform(-max_angle, max_angle)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # 生成旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行旋转（使用边缘填充）
    rotated = cv2.warpAffine(
        img, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def flip_image(img):
    return cv2.flip(img, 1)  # 1表示水平翻转


def translate_image(img, max_translate=0.2):
    height, width = img.shape[:2]
    tx = random.uniform(-max_translate, max_translate) * width
    ty = random.uniform(-max_translate, max_translate) * height

    # 生成平移矩阵
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    # 执行平移（使用边缘填充）
    translated = cv2.warpAffine(
        img, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )
    return translated


def add_noise_to_image(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img


if __name__ == "__main__":
    # 输入输出目录配置
    input_dir = r"data1\video"  # 原始图片目录
    output_dir = r"data1\picture_augmented"  # 增强后图片保存目录

    # 执行数据增强（每个原始图片生成5个增强版本）
    augment_images(input_dir, output_dir, num_augmentations=5)
    print("图片数据增强完成！增强后的图片已保存至：", output_dir)