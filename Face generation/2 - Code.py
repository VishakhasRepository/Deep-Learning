import urllib.request

# Downloading the dataset
url = 'https://www.kaggle.com/jessicali9530/celeba-dataset'
urllib.request.urlretrieve(url, "celeba_dataset.zip")

# Unzipping the dataset
!unzip celeba_dataset.zip

from PIL import Image
import numpy as np
import os

# Creating a list of image file names
image_file_names = os.listdir('img_align_celeba')

# Resizing the images and converting them to numpy arrays
images = []
for image_file_name in image_file_names:
    image = Image.open(os.path.join('img_align_celeba', image_file_name))
    image = image.resize((64, 64))
    image = np.array(image)
    images.append(image)

# Normalizing the images
images = np.array(images)
images = images / 255.0

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model

# Defining the generator
generator_input = Input(shape=(100,))
generator = Dense(256)(generator_input)
generator = LeakyReLU(alpha=0.2)(generator)
generator = Reshape((4, 4, 16))(generator)
generator = Conv2DTranspose(16, kernel_size=(4,4), strides=(2,2), padding='same')(generator)
generator = LeakyReLU(alpha=0.2)(generator)
generator = Conv2DTranspose(32, kernel_size=(4,4), strides=(2,2), padding='same')(generator)
generator = LeakyReLU(alpha=0.2)(generator)
generator = Conv2DTranspose(3, kernel_size=(4,4), strides=(2,2), padding='same', activation='sigmoid')(generator)
generator_model = Model(generator_input, generator)

# Defining the discriminator
discriminator_input = Input(shape=(64, 64, 3))
discriminator = Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same')(discriminator_input)
discriminator = LeakyReLU(alpha=0.2)(discriminator)
discriminator = Conv2D(16, kernel_size=(4,4), strides=(2,2), padding='same')(discriminator)
discriminator = LeakyReLU(alpha=0.2)(discriminator)
discriminator = Flatten()(discriminator)
discriminator = Dense(1, activation='sigmoid')(discriminator)
discriminator_model = Model(discriminator_input, discriminator)

# Defining the GAN
gan_input = Input(shape=(100,))
gan_output = discriminator_model(generator_model(gan_input))
gan_model = Model(gan_input, gan_output)

# Compiling the models
discriminator_model.compile(loss='binary_crossentropy', optimizer='adam')
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# Training the model
epochs = 100
batch_size = 128
for epoch in range(epochs):
    for i in range(len(images)//batch_size):
        # Training the discriminator
        discriminator_model.trainable = True
        discriminator_batch = images[i*batch_size:(i+1)*batch_size]
        noise = np.random.normal(size=(batch_size, 100))
        fake_images = generator_model.predict(noise)
        discriminator_loss_real = discriminator_model.train_on_batch

# Define optimizer and loss function
optimizer = Adam(lr=0.0002, beta_1=0.5)
loss_fn = BinaryCrossentropy(from_logits=True)

# Define checkpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint(
    'face_generator_best.h5', monitor='val_loss', save_best_only=True)

# Compile the generator model
generator.compile(optimizer=optimizer, loss=loss_fn)

# Train the generator
history = generator.fit(train_dataset, epochs=10, validation_data=val_dataset,
                        callbacks=[checkpoint_callback])

# Load the best generator model
generator = load_model('face_generator_best.h5')

# Generate faces using the generator model
generated_faces = generator.predict(noise)

# Rescale the generated faces from [-1, 1] to [0, 1]
generated_faces = (generated_faces + 1) / 2.0

# Visualize the generated faces
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_faces[i])
    ax.set_title(f'Face {i+1}')
plt.show()
