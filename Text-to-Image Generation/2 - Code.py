import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

batch_size = 128
learning_rate = 0.0002
num_epochs = 50
latent_size = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Section 5: Train the Model

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
lr = 0.0002
batch_size = 64
num_epochs = 50
beta1 = 0.5
beta2 = 0.999

# Initialize generator and discriminator models
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Initialize loss functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Create dataloader for training images
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, captions) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones((imgs.size(0), 1)).to(device)
        fake = torch.zeros((imgs.size(0), 1)).to(device)

        # Configure input
        real_imgs = imgs.to(device)
        captions = captions.to(device)
        z = torch.randn((imgs.shape[0], latent_dim)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_imgs = generator(z, captions)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, captions), valid)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z, captions)

        # Measure pixelwise loss between real and generated images
        pixel_loss = pixelwise_loss(gen_imgs, real_imgs)

        # Measure discriminator's ability to classify generated samples
        validity = discriminator(gen_imgs, captions)
        g_loss = adversarial_loss(validity, valid) + lambda_pixel * pixel_loss

        g_loss.backward()
        optimizer_G.step()

        # Print training progress
        batches_done = epoch * len(dataloader) + i
        if batches_done % print_interval == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

        # Save generated images at specified interval
        if batches_done % save_interval == 0:
            save_image(gen_imgs.data[:25], f"{output_dir}/images/{batches_done}.png", nrow=5, normalize=True)

# Section 5: Train the Model

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
lr = 0.0002
batch_size = 64
num_epochs = 50
beta1 = 0.5
beta2 = 0.999

# Initialize generator and discriminator models
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Initialize loss functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Create dataloader for training images
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, captions) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones((imgs.size(0), 1)).to(device)
        fake = torch.zeros((imgs.size(0), 1)).to(device)

        # Configure input
        real_imgs = imgs.to(device)
        captions = captions.to(device)
        z = torch.randn((imgs.shape[0], latent_dim)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_imgs = generator(z, captions)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, captions), valid)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z, captions)

        # Measure pixelwise loss between real and generated images
        pixel_loss = pixelwise_loss(gen_imgs, real_imgs)

        # Measure discriminator's ability to classify generated samples
        validity = discriminator(gen_imgs, captions)
        g_loss = adversarial_loss(validity, valid) + lambda_pixel * pixel_loss

        g_loss.backward()
        optimizer_G.step()

        # Print training progress
        batches_done = epoch * len(dataloader) + i
        if batches_done % print_interval == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

        # Save generated images at specified interval
        if batches_done % save_interval == 0:
            save_image(gen_imgs.data[:25], f"{output_dir}/images/{batches_done}.png", nrow=5, normalize=True)

# 6. Evaluation Metrics

from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

# Load test images
test_images = ['test1.png', 'test2.png', 'test3.png', 'test4.png', 'test5.png']
fig, axs = plt.subplots(nrows=len(test_images), ncols=2, figsize=(10, 20))

# Generate images and calculate evaluation metrics
for i, img_path in enumerate(test_images):
    # Load text description and generate image
    text = test_descriptions[i]
    generated_image = generate_image(text, decoder_model, image_size)

    # Load ground truth image
    true_image = img_to_array(load_img('test_images/'+img_path, target_size=(image_size, image_size)))

    # Calculate SSIM and PSNR
    ssim = structural_similarity(true_image, generated_image, multichannel=True)
    psnr = peak_signal_noise_ratio(true_image, generated_image)

    # Plot images and evaluation metrics
    axs[i, 0].imshow(generated_image.astype(np.uint8))
    axs[i, 0].set_title('Generated Image')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(true_image.astype(np.uint8))
    axs[i, 1].set_title('True Image\nSSIM: {:.3f}\nPSNR: {:.3f}'.format(ssim, psnr))
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

# 7. Results

# Save model weights
encoder_model.save_weights('encoder_model_weights.h5')
decoder_model.save_weights('decoder_model_weights.h5')

# Generate and save example images
for i, text in enumerate(example_descriptions):
    generated_image = generate_image(text, decoder_model, image_size)
    plt.imshow(generated_image.astype(np.uint8))
    plt.axis('off')
    plt.savefig('example_{}.png'.format(i+1))

print('Model trained successfully. Example images saved to disk.')
