import pandas as pd
import re

# Function to extract image names from log.txt and remove the .jpg extension
def extract_image_names(log_path):
    image_names = []
    with open(log_path, 'r') as file:
        log_content = file.readlines()
    for line in log_content:
        match = re.search(r'No hands detected in image (\S+)', line)
        if match:
            image_name = match.group(1).strip()  # Remove any extra whitespace
            image_name_without_ext = image_name.replace('.jpg', '')  # Remove .jpg extension
            image_names.append(image_name_without_ext)
            print(f"Extracted image name (without .jpg): {image_name_without_ext}")  # Debug information
    return image_names

# Define the paths for the log file and the CSV file
log_path = r'.\data1\log.txt'
csv_path = r'.\data1\find_hand.csv'

# Extract the list of image IDs to be removed
images_to_remove = extract_image_names(log_path)

# Check the extracted image name list
print(f"Images to remove: {images_to_remove}")

# Read the CSV file, skipping corrupted lines
try:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Check the column names of the CSV file
if 'Image_ID' not in df.columns:
    print("CSV file does not contain an 'Image_ID' column.")
    exit()

# Print the first few rows of the CSV file to verify column names and data
print("First few rows of the CSV file:")
print(df.head())

# Verify whether the extracted image names exist in the CSV file
existing_images_in_csv = set(df['Image_ID'])
non_existent_images = [img for img in images_to_remove if img not in existing_images_in_csv]

if non_existent_images:
    print(f"The following images were not found in the CSV file: {non_existent_images}")
else:
    print("All extracted images are present in the CSV file.")

# Remove rows from the DataFrame where Image_ID is in images_to_remove
df_filtered = df[~df['Image_ID'].isin(images_to_remove)]

# Print the number of rows before and after filtering
print(f"Original DataFrame shape: {df.shape}")
print(f"Filtered DataFrame shape: {df_filtered.shape}")

# Save the filtered data back to a new CSV file
output_csv_path = r'.\data1\find.csv'
df_filtered.to_csv(output_csv_path, index=False)

print(f"Filtered CSV has been saved to {output_csv_path}")
