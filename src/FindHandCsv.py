import os  # Import the os module for path operations
import cv2
import mediapipe as mp
import csv

# Core module for data preprocessing in the hand gesture recognition system.
# It primarily extracts hand keypoint coordinates from gesture images and generates a structured dataset,
# providing input data for subsequent machine learning model training.

# Initialize the Mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the drawing utility
mp_drawing = mp.solutions.drawing_utils

def process_hand_gesture_from_images(input_folder, output_file):

    # Ensure the output file directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a CSV file and write the header
    header = ["Image_ID"]  # Modify the first column as the image ID
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

    # Add labels column
    header.append("Label")  # Add label column

    # Create a CSV file and write the header
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Get all image files from the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_files.sort()  # Sort by filename to ensure correct processing order

    # Process each image
    for image_file in image_files:
        # Get the image filename as the label
        image_name = os.path.splitext(image_file)[0]  # Get the image name (without extension)

        # Construct the image path and read the image
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error opening image file {image_file}")
            continue

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform hand keypoint detection
        results = hands.process(rgb_image)

        row = []
        # Use image ID as Image_ID
        image_id = image_name
        row.append(image_id)

        # Extract gesture information
        gesture = "Detected"  # Gesture classification can be customized as needed

        # Check if any hands are detected
        if not results.multi_hand_landmarks:
            print(f"No hands detected in image {image_file}")
            row += [0] * 126  # Fill coordinates with zeros for both hands
            row.append(image_name)
            # Export gesture and keypoint data to the CSV file
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            continue

        # Iterate through detected hands
        for hand_num, landmarks in enumerate(results.multi_hand_landmarks, 1):
            # Extract gesture information and keypoint data
            landmark_coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

            # Split coordinates into separate columns
            for coord in landmark_coords:
                row.append(coord[0])  # x
                row.append(coord[1])  # y
                row.append(coord[2])  # z

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        # If only one hand is detected, fill the second hand's coordinates with zeros
        if len(results.multi_hand_landmarks) == 1:
            row += [0] * 63  # Fill the second hand's 63 coordinates with zeros

        # If a second hand is detected, coordinates are recorded correctly
        elif len(results.multi_hand_landmarks) == 2:
            pass  # The second hand is already recorded, do nothing

        # Append image label at the end of the row
        row.append(image_name)

        # Export gesture and keypoint data to the CSV file
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # Optional: Display the recognition result
        cv2.imshow("Hand Gesture Recognition", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder = r".\data1\picture_augmented"  # Input image folder path
    output_file = r".\data1\find_hand.csv"  # Output CSV file path
    process_hand_gesture_from_images(input_folder, output_file)
