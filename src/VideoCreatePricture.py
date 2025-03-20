import cv2
import os

# Extract key frames from videos, mainly to solve the conversion from video data to image data, which is a common preprocessing step in computer vision projects (such as gesture recognition)

# Video folder path
video_folder = "./data/video"

# Output folder path for frames
output_folder = "./data/picture"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all video files in the video folder
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Number of frames to extract per second
fps_target = 30

# Dictionary to prevent name duplication
name_counter = {}

# Iterate through each video file
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file}")
        continue

    # Get the original frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure frame_interval is not 0 by preventing fps_target from being greater than the original fps
    frame_interval = max(int(fps / fps_target), 1)

    frame_count = 0
    frame_num = 0

    # Get the base name of the video (excluding the extension)
    video_name = os.path.splitext(video_file)[0]

    # Initialize the counter
    if video_name not in name_counter:
        name_counter[video_name] = 0

    # Iterate through each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Extract a frame every certain number of frames
        if frame_count % frame_interval == 0:
            name_counter[video_name] += 1
            # Generate the frame filename in the format video_name_001.jpg, video_name_002.jpg, etc.
            output_filename = os.path.join(output_folder, f"{video_name}_{name_counter[video_name]:03d}.jpg")
            cv2.imwrite(output_filename, frame)

    # Release the video object
    cap.release()

print("Frame extraction completed")
