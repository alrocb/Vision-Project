import cv2
import numpy as np
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Sigmoid activation function for the output to range between 0 and 1 (for image pixels)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
# Load the trained autoencoder
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoder.pth'))
autoencoder.eval()

# Function to compute Fred for a patch
def compute_fred(original_patch, reconstructed_patch):
    # Convert patches to HSV color space
    original_hsv = cv2.cvtColor(original_patch, cv2.COLOR_BGR2HSV)
    reconstructed_hsv = cv2.cvtColor(reconstructed_patch, cv2.COLOR_BGR2HSV)
    
    # Define hue range for red-like pixels
    lower_red = np.array([-20, 100, 100])
    upper_red = np.array([20, 255, 255])
    
    # Create masks for red-like pixels
    original_mask = cv2.inRange(original_hsv, lower_red, upper_red)
    reconstructed_mask = cv2.inRange(reconstructed_hsv, lower_red, upper_red)
    
    # Compute the fraction of red-like pixels lost (Fred)
    original_red_pixels = np.sum(original_mask > 0)
    reconstructed_red_pixels = np.sum(reconstructed_mask > 0)
    fred = reconstructed_red_pixels / original_red_pixels
    
    return fred

# Function to diagnose H. pylori presence based on precomputed patches
import torch
import matplotlib.pyplot as plt

# Function to diagnose H. pylori presence and visualize original and reconstructed patches
def diagnose_hpylori_presence(patch_folder, threshold=0.1):
    # List all patch files in the folder
    patch_files = [os.path.join(patch_folder, file) for file in os.listdir(patch_folder) if file.endswith('.png')]
    
    # Count patches with Fred > 1
    positive_patches = 0
    total_patches = len(patch_files)
    
    original_patches = []
    reconstructed_patches = []
    diagnoses = []
    
    for patch_file in patch_files:
        # Load original patch as a numpy array and convert it to a PyTorch tensor
        original_patch = cv2.imread(patch_file)
        original_patch = original_patch.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        original_patch = np.transpose(original_patch, (2, 0, 1))  # Change data layout from HWC to CHW
        original_patch_tensor = torch.from_numpy(original_patch).unsqueeze(0)  # Add batch dimension
        
        # Pass the patch through the autoencoder
        with torch.no_grad():
            reconstructed_patch_tensor = autoencoder(original_patch_tensor)
            reconstructed_patch = reconstructed_patch_tensor.numpy()[0]  # Remove batch dimension
        
        # Compute Fred for the patch
        fred = compute_fred(original_patch_tensor.numpy()[0].transpose(1, 2, 0), reconstructed_patch.transpose(1, 2, 0))
        
        # Count positive patches
        if fred > 1:
            positive_patches += 1
            diagnoses.append("H. pylori present")
        else:
            diagnoses.append("No evidence of H. pylori")
        
        # Store original and reconstructed patches
        original_patches.append(original_patch)
        reconstructed_patches.append(reconstructed_patch)
    
    # Calculate the percentage of positive patches
    positive_percentage = positive_patches / total_patches
    
    # Diagnose H. pylori presence based on the threshold
    if positive_percentage > threshold:
        overall_diagnosis = "H. pylori is present in the histological image."
    else:
        overall_diagnosis = "No evidence of H. pylori in the histological image."
    
    return original_patches, reconstructed_patches, diagnoses, overall_diagnosis

# Example usage: provide the folder path containing the precomputed patches
patch_folder = '/home/sisard/Documents/universitat/year_3/semester_1/Computer_Vision/Medical_images/Project/model/test_metric'
original_patches, reconstructed_patches, diagnoses, overall_diagnosis = diagnose_hpylori_presence(patch_folder, threshold=0.1)

# Visualize original and reconstructed patches along with the diagnosis
for i in range(len(original_patches)):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_patches[i].transpose(1, 2, 0))
    axes[0].set_title('Original Patch')
    axes[0].axis('off')
    axes[1].imshow(reconstructed_patches[i].transpose(1, 2, 0))
    axes[1].set_title('Reconstructed Patch')
    axes[1].axis('off')
    plt.suptitle(diagnoses[i])
    plt.show()

print("Overall Diagnosis:", overall_diagnosis)


















