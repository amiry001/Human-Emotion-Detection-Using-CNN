import os
import random
from PIL import Image
import numpy as np

# Set the file paths for the dataset and output directories
dataset_dir = "D:\waveletsegmentation"
output_dir = "D:\waveletsegmentationvalidation"

# Define the emotions to include and the number of images to select from each emotion
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_images_per_emotion = 600

# Function to replace invalid characters in a filename
def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# Loop over each emotion and select a random sample of images to preprocess
for emotion in emotions:
    emotion_dir = os.path.join(dataset_dir, emotion)
    files = os.listdir(emotion_dir)
    sample = random.sample(files, num_images_per_emotion)

    # Create the emotion folder in the output directory if it doesn't exist
    output_emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_dir, exist_ok=True)

    # Loop over each selected image and preprocess it
    for filename in sample:
        filepath = os.path.join(emotion_dir, filename)

        # Load and preprocess the image
        image = Image.open(filepath)
        image = image.resize((256, 256))  # Resize the image
        image = np.array(image)  # Convert the image to a numpy array
        # Apply any additional preprocessing steps here

        # Sanitize the filename to replace invalid characters
        sanitized_filename = sanitize_filename(filename)

        # Save the preprocessed image in the appropriate emotion folder
        output_path = os.path.join(output_emotion_dir, sanitized_filename)

        # Save the preprocessed image
        Image.fromarray(image).save(output_path)

        # Remove the image from the HOGtest folder
        #os.remove(filepath)
