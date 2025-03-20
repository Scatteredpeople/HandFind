import re
import os

# Function to extract image names from log.txt
def extract_image_names(log_path):
    image_names = []
    with open(log_path, 'r') as file:
        log_content = file.readlines()
    for line in log_content:
        match = re.search(r'No hands detected in image (\S+)', line)
        if match:
            image_name = match.group(1).strip()  # Remove any extra spaces
            image_names.append(image_name)  # Keep the .jpg extension
            print(f"Extracted image name: {image_name}")  # Debug information
    return image_names

# Define file paths
log_path = r'.\data1\log.txt'
pictures_folder = r'.\data1\picture_augmented'

# Extract the list of image IDs to be deleted
images_to_remove = extract_image_names(log_path)

# Check the extracted image name list
print(f"Images to remove: {images_to_remove}")

# Delete corresponding image files from the pictures folder
for image_name in images_to_remove:
    image_file_path = os.path.join(pictures_folder, image_name)
    try:
        os.remove(image_file_path)
        print(f"Deleted image file: {image_file_path}")
    except FileNotFoundError:
        print(f"Image file not found: {image_file_path}")
    except Exception as e:
        print(f"Error deleting image file {image_file_path}: {e}")
