from sklearn.model_selection import train_test_split
from keras import models, layers, Input, Model
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob
import cv2
import os

# This is the first approach I will be trying out. The goal is to build an object detection model for strawberries.

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("\nAll physical devices:", physical_devices, "\n")

def load_and_preprocess_images(directory, target_size=(64, 64), augment=False, rotate_degrees=[90],
                               flips=[0], scale=1.0, colors=['gray']):
    
    """
    Load images from a directory, convert them to specified color schemes, resize, normalize pixel values, optionally apply augmentations 
    including multiple flip orientations, and perform corner detection or template matching.
    
    Parameters:
    - directory: Path to the directory containing images.
    - target_size: A tuple (width, height) representing the target image size.
    - augment: Boolean indicating whether to apply augmentations.
    - rotate_degrees: List of degrees to rotate the image. Applied if augment is True.
    - flips: List of orientations to flip the image. Can include 0 (vertical), 1 (horizontal), or -1 (both).
    - scale: Scaling factor for the image, with 1.0 meaning no scaling. Applied if augment is True.
    - colors: List of color schemes to convert images into. Supports 'bgr', 'gray', and 'hsv'.
    
    Returns:
    - A list of NumPy arrays containing the processed, augmented, and analyzed image data.
    - The number of channels in the images
    """

    images = []
    image_extensions = ['jpg', 'jpeg', 'png']
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{extension}')))
    
    for img_path in image_paths:
        original_img = cv2.imread(img_path)

        for color in colors:
            if color == 'gray':
                img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                num_channels = 1
            elif color == 'hsv':
                img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
                num_channels = 3
            else:  # Default to BGR
                img = original_img
                num_channels = 3
            
            # Resize and normalize the original image
            img = cv2.resize(img, target_size)
            img = img / 255.0
            
            # Add the original image to the list
            images.append(img)

            if augment:
                for degree in rotate_degrees:
                    for flip in flips:
                        augmented_img = img.copy()

                        # Rotate image
                        if degree != 0:
                            M = cv2.getRotationMatrix2D((target_size[0] / 2, target_size[1] / 2), degree, scale)
                            augmented_img = cv2.warpAffine(augmented_img, M, target_size)
                            
                        # Apply flip
                        augmented_img = cv2.flip(augmented_img, flip)

                    # Add the augmented image to the list
                    images.append(augmented_img)

    # Convert the list of images to a NumPy array
    images = np.array(images)
    return images, num_channels

# Load the unlabeled strawDI images dataset
path = 'C:/Users/juanm/Documents/Capstone/datasets'
images, num_channels = load_and_preprocess_images(path+"/straw_di", target_size=(64, 64), augment=False,
                                                  rotate_degrees=[90], flips=[0], scale=1.0, colors=['bgr'])

print(f"Shape of the images: {images.shape}")

# Reshape
img_width = images.shape[1]
img_height = images.shape[2]
images = images.reshape(images.shape[0], img_width, img_height, num_channels)
input_shape = images.shape[1:]
cmap = 'gray' if num_channels == 1 else None

# Plot some images
plt.figure(1)
plt.subplot(221)
plt.imshow(images[42][:,:,0], cmap=cmap)

plt.subplot(222)
plt.imshow(images[420][:,:,0], cmap=cmap)
