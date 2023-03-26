# Import necessary libraries and modules
import os
import numpy as np
from PIL import Image

# Define path to data directory
data_dir = 'path/to/data/directory/'

# Define function to preprocess and load data
def load_data(data_dir, image_size):
    images = []
    for img_file in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, img_file))
        img = img.resize((image_size, image_size))
        img = np.array(img) / 255.0
        images.append(img)
    return np.array(images)

# Load and preprocess data
image_size = 256
data = load_data(data_dir, image_size)

# Import necessary libraries and modules
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

# Define generator model architecture
def make_generator_model():
    # Define input layer
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    
    # Define encoder layers
    e1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)
    e1 = LeakyReLU()(e1)
    e2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU()(e2)
    e3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU()(e3)
    e4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU()(e4)
    e5 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU()(e5)
    e6 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e5)
    e6 = BatchNormalization()(e6)
    e6 = LeakyReLU()(e6)
    e7 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e6)
    e7 = BatchNormalization()(e7)
    e7 = LeakyReLU()(e7)
    e8 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)(e7)
    e8 = BatchNormalization()(e8)
    e8 = LeakyReLU()(e8)

    # Define decoder layers
    dec1 = UpSampling2D(size=(2, 2))(encoder_output)
    dec1 = Conv2D(512, 3, padding='same', activation='relu')(dec1)
    dec1 = BatchNormalization()(dec1)

    dec2 = Concatenate()([dec1, skip2])
    dec2 = UpSampling2D(size=(2, 2))(dec2)
    dec2 = Conv2D(256, 3, padding='same', activation='relu')(dec2)
    dec2 = BatchNormalization()(dec2)

    dec3 = Concatenate()([dec2, skip1])
    dec3 = UpSampling2D(size=(2, 2))(dec3)
    dec3 = Conv2D(128, 3, padding='same', activation='relu')(dec3)
    dec3 = BatchNormalization()(dec3)

    dec4 = UpSampling2D(size=(2, 2))(dec3)
    dec4 = Conv2D(64, 3, padding='same', activation='relu')(dec4)
    dec4 = BatchNormalization()(dec4)

    # Output layer
    output = Conv2D(3, 1, padding='same', activation='tanh')(dec4)

    # Define the generator model
    generator = Model(input, output, name='generator')


