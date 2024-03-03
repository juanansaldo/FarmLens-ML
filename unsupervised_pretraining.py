from keras import models, layers, Input, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob
import cv2
import os

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("\nAll physical devices:", physical_devices, "\n")

def load_images_from_directory(directory, target_size=(64, 64)):
    """
    Load images from a directory, convert them to a specified size, and normalize pixel values.
    
    Parameters:
    - directory: Path to the directory containing images.
    - target_size: A tuple (width, height) representing the target image size.
    
    Returns:
    - A NumPy array containing the processed image data.
    """
    images = []
    # Supported image formats
    image_extensions = ['jpg', 'jpeg', 'png']
    # Create a list of all image paths in the directory
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{extension}')))
    
    for img_path in image_paths:
        # Load the image in BGR format
        img = cv2.imread(img_path)
        # Convert the image from BGR to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize the image to the target size
        img = cv2.resize(img, target_size)
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        images.append(img)
    
    # Convert the list of images to a NumPy array
    images_array = np.array(images)
    return images_array

# Load the data
directory_path = 'C:/Users/juanm/Documents/Capstone/datasets/straw_di'
images = load_images_from_directory(directory_path, target_size=(64, 64))

# Normalize
images = images.astype('float32') / 255

# Reshape
img_width = images.shape[1]
img_height = images.shape[2]
num_channels = 1
images = images.reshape(images.shape[0], img_width, img_height, num_channels)
input_shape = (img_height, img_width, num_channels)

# Plot some images
plt.figure(1)
plt.subplot(221)
plt.imshow(images[42][:,:,0])

plt.subplot(222)
plt.imshow(images[420][:,:,0])

# Train test split
X_train, X_test = train_test_split(images, test_size=0.3, random_state=0)

#
# ===Build the model===
#

# Delete old models
K.clear_session()

# ===Encoder===
latent_dim = 2

input_img = Input(shape=input_shape, name='encoder_input')
x = layers.Conv2D(32, 3, padding='same', activation='elu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='elu', strides=2)(x)
x = layers.Conv2D(64, 3, padding='same', activation='elu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='elu')(x)

conv_shape = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='elu')(x)

# Two outputs, for latent mean and log variance (std)
# Use these to sample random variables in latent space to which the inputs are mapped
z_mu = layers.Dense(latent_dim, name='latend_mu')(x) # Mean values of encoded input
z_sigma = layers.Dense(latent_dim, name='latent_sigma')(x) # std of encoded input

# Reparameterization trick
# Define sampling function to sample from the distribution.
# Reparameterize sample based on the process defined by Gunderson and Huang
# into the shape of: mu + sigma squared x eps.
# This is to allow gradient descent estimation accurately.

def sample_z(args):
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps

# sample vector from the latent distribution
# z is the lambda custom layer we are adding for gradient descent calculations
# using mu and variance (sigma)
z = layers.Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])

# Z (lambda layer) will be the last layer in the encoder
# Define and summarize encoder model.
encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
print(encoder.summary())

# ===Decoder===
# decoder takes the latent vector as input
decoder_input = Input(shape=(latent_dim,), name='decoder_input')

# Need to start with a shape that can be remapped to original shape as
# we want our final output to be the same shape as the original input.
# Add dense layer with dimensions that can be reshaped to desired output shape
x = layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='elu')(decoder_input)

# reshape to the data of last conv. layer in the encoder, so we can upscale
# back to original shape
x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

# use Conv2DTranspose to reverse the conv layers defind in the encoder
x = layers.Conv2DTranspose(32, 3, padding='same', activation='elu', strides=2)(x)

# using sigmoid activation
x = layers.Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder')(x)

# Define and summarize decoder model
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# apply the decoder to the latent sample
z_decoded = decoder(z)

# ===Define custom loss function===

# VAE is trained using two loss functions' reconstruction loss, and KL divergence
# Let us add a class to define a custom layer with loss
class CustomLayer(layers.Layer):

    def vae_loss(self, x, z_decoded, z_mu, z_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_sigma = inputs[3]

        loss = self.vae_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution
y = CustomLayer()([input_img, z_decoded, z_mu, z_sigma])

# y is basically the original image after encoding input img to mu, sigma, z,
# and decoding sampled z values.
# This will be used as output for vae

# ===VAE===
vae = Model(input_img, y, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
vae.fit(X_train, None, epochs=10, batch_size=32, validation_split=0.2)

# ===Visualize results===
# Visualize inputs mapped to the latent space
# Remember that we have encoded inputs to latent space dimension = 2.

# Extract z_mu -> first parameter in the result of encoder prediction representating mean
mu, _, _ = encoder.predict(X_test)

# plot dim1 and dim2 for mu
plt.figure(figsize=(12, 12))
plt.scatter(mu[:, 0], mu[:, 1], cmap='brg')
plt.xlabel('dim1')
plt.ylabel('dim2')
plt.colorbar()
plt.show();

# Visualize images
# Single decoded images with random input latent vector (of size 1x2)
# latent space range is about -5 to 5, so pick random values within this range
# Try starting with -1, 1 and slowly go up to -1.5,1.5 and see how it morphs from
# one image to the other
sample_vector = np.array([[1,3]])
decoded_example = decoder.predict(sample_vector)
decoded_example_reshaped = decoded_example.reshape(img_width, img_height)
plt.imshow(decoded_example_reshaped)

# Let us automate this process by generating multiple images and plotting
# Use decoder to generate images by tweaking latent variables from the latent space
# Create a grid of defined size with zeros.
# Take sample from some defined linear space. In this example range [-4, 4]
# Feed it to the decoder and update zeros in the figure with output

n = 20
figure = np.zeros((img_width * n, img_height * n, num_channels))

# Create a grid of latent variables, to be provided as inputs to decoder. Predict
# Creating vectors within range -5 to 5 as that seems to be the range in latent space
grid_x = np.linspace(-5, 5, n)
grid_y = np.linspace(-5, 5, n)[::-1]

# decoder for each square in the grid
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = digit

plt.figure(figsize=(12, 12))
# Reshape for visualization
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1]))

plt.imshow(figure, cmap='gnuplot2')
plt.show();
