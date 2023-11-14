import torch

from torchvision import transforms

import matplotlib.pyplot as plt
from model_def import Generator
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for generating images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Number of images to generate
num_images = 10

# Load the generator model and move it to the appropriate device
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()  # Set the model to evaluation mode

# Generate random noise on the same device as the generator
noise_dim = 100
batch_size = 10  # Number of samples to generate
random_noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

# Generate samples from the generator
with torch.no_grad():
    generated_samples = generator(random_noise)

# Convert PyTorch tensor to NumPy array
generated_images_np = generated_samples.detach().cpu().numpy()

# Visualize the generated samples
fig, axs = plt.subplots(1, batch_size, figsize=(15, 3))
for i in range(batch_size):
    axs[i].imshow(np.transpose(generated_images_np[i], (1, 2, 0)))
    axs[i].axis('off')

plt.show()
