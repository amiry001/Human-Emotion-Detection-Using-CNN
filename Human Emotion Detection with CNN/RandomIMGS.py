import os
import random
from PIL import Image
import numpy as np

# Set the file paths for the dataset and output directories
dataset_dir = "D:/preproccess"
output_dir = "D:/Gpre"

# Define the emotions to include and the number of images to select from each emotion
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_images_per_emotion = 160

# Loop over each emotion and select a random sample of images to preprocess
for emotion in emotions:
    emotion_dir = os.path.join(dataset_dir, emotion)
    files = os.listdir(emotion_dir)
    sample = random.sample(files, num_images_per_emotion)

    # Loop over each selected image and preprocess it
    for filename in sample:
        filepath = os.path.join(emotion_dir, filename)

        # Load and preprocess the image
        image = Image.open(filepath)
        image = image.resize((256, 256))  # Resize the image
        image = np.array(image)  # Convert the image to a numpy array
        # Apply any additional preprocessing steps here

        # Remove the emotion prefix from the original filename
        original_filename = filename[len(emotion) + 1:]  # +1 to remove the underscore

        # Label the image with the emotion class
        labeled_filename = f"{emotion}_{original_filename}"

        # Save the preprocessed image in the appropriate emotion folder
        output_path = os.path.join(output_dir, emotion, labeled_filename)

        # Save the preprocessed image
        Image.fromarray(image).save(output_path)
