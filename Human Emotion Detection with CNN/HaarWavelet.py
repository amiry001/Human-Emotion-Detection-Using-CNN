import pywt
import numpy as np
from PIL import Image
import os

# Set the path to the dataset directory
dataset_path = 'C:/Users/Fatima Amiri/Desktop/black/Wavelet.jpeg'
path2 = 'C:/Users/Fatima Amiri/Desktop/black/Waveletseg.jpg'

# Create a list of all image file names in the dataset directory
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# Create an empty list to store the transformed coefficients for each image
coeffs_list = [] 

# Loop over each image file in the dataset
for image_file in image_files:
    # Load the image as a NumPy array
    image = np.array(Image.open(image_file).convert('L'))

    # Perform a 2D discrete wavelet transform
    coeffs = pywt.dwt2(image, 'haar')

    # Append the transformed coefficients to the list
    coeffs_list.append(coeffs)

# Save the transformed coefficients as a new dataset of images
for i, coeffs in enumerate(coeffs_list):
    coeffs_image = np.stack([coeffs[0], coeffs[1][0], coeffs[1][1]], axis=-1)
    Image.fromarray(coeffs_image.astype('uint8')).save(os.path.join(path2, f'coeffs_{i}.jpg'))