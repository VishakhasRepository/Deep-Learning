# Section 1: Importing Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# Section 2: Helper Functions
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(img, title=None):
    out = np.squeeze(img.numpy(), axis=0)
    out = out.clip(0, 1)
    plt.imshow(out)
    if title:
        plt.title(title)
    plt.imshow(out)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load the VGG model and access the intermediate layers
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    # Create the model
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """ Calculates the gram matrix of the input tensor"""
    # Flatten the tensor
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]

    # Calculate the gram matrix
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Section 3: Loading and Preparing Images
content_path = 'path/to/content/image.jpg'
style_path = 'path/to/style/image.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

import tensorflow as tf
import tensorflow_hub as hub

# Load the VGG19 model for style extraction
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Define the style and content layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']

# Define the Model for style transfer
def style_transfer_model(style_layers, content_layers):
    # Set up the inputs
    style_input = tf.keras.layers.Input(shape=IMAGE_SIZE+(3,), name='style_image')
    content_input = tf.keras.layers.Input(shape=IMAGE_SIZE+(3,), name='content_image')
    
    # Load the VGG19 model and set the layers to be non-trainable
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Get the outputs from the style layers and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    
    # Build the model
    return tf.keras.models.Model(inputs=[style_input, content_input], outputs=model_outputs)

# Build the style transfer model
model = style_transfer_model(style_layers, content_layers)

# Define the optimizer and the loss function
optimizer = tf.optimizers.Adam(learning_rate=0.02)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# Define the number of epochs
epochs = 10

# Train the model
for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch + 1, epochs))
    for image, style in train_dataset:
        with tf.GradientTape() as tape:
            stylized_image = transformer(image)
            loss = mse_loss_fn(style, stylized_image)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # Save the model
    if (epoch + 1) % 5 == 0:
        transformer.save_weights("models/epoch-{}.ckpt".format(epoch + 1))

# Load the saved model
transformer.load_weights("models/epoch-{}.ckpt".format(epochs))

# Define the function for stylizing an image
def stylize_image(image_path):
    # Load the image
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)

    # Stylize the image
    stylized_image = transformer(image)

    # Save the stylized image
    stylized_image = tf.squeeze(stylized_image, axis=0)
    stylized_image = tf.clip_by_value(stylized_image, 0, 255)
    stylized_image = tf.cast(stylized_image, tf.uint8)
    output_image_path = "outputs/stylized-{}".format(os.path.basename(image_path))
    save_image(output_image_path, stylized_image.numpy())

# Test the model on a sample image
stylize_image("images/butterfly.jpg")
