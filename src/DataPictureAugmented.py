import os
import cv2
import numpy as np
import random
import math


def augment_images(input_dir, output_dir, num_augmentations=5):
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the original image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Save the original image (optional)
            base_name, ext = os.path.splitext(filename)
            # cv2.imwrite(os.path.join(output_dir, f"{base_name}_original{ext}"), img)

            # Generate augmented images
            for i in range(num_augmentations):
                aug_img = img.copy()

                # Randomly select augmentation type
                augmentation_type = random.choice(['rotate', 'flip', 'translate', 'noise'])

                if augmentation_type == 'rotate':
                    aug_img = rotate_image(aug_img)
                elif augmentation_type == 'flip':
                    aug_img = flip_image(aug_img)
                elif augmentation_type == 'translate':
                    aug_img = translate_image(aug_img)
                elif augmentation_type == 'noise':
                    aug_img = add_noise_to_image(aug_img)

                # Save the augmented image
                output_path = os.path.join(
                    output_dir,
                    f"{base_name}_aug{i}_{augmentation_type}{ext}"
                )
                cv2.imwrite(output_path, aug_img)


def rotate_image(img, max_angle=45):
    angle = random.uniform(-max_angle, max_angle)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # Generate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation (using edge padding)
    rotated = cv2.warpAffine(
        img, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def flip_image(img):
    return cv2.flip(img, 1)  # 1 indicates horizontal flip


def translate_image(img, max_translate=0.2):
    height, width = img.shape[:2]
    tx = random.uniform(-max_translate, max_translate) * width
    ty = random.uniform(-max_translate, max_translate) * height

    # Generate translation matrix
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    # Perform translation (using edge padding)
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
    # Configure input and output directories
    input_dir = r"data1\video"  # Directory containing original images
    output_dir = r"data1\picture_augmented"  # Directory to save augmented images

    # Perform data augmentation (generate 5 augmented versions per original image)
    augment_images(input_dir, output_dir, num_augmentations=5)
    print("Image data augmentation completed! Augmented images have been saved to:", output_dir)
