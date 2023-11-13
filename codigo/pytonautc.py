'''
############################################
ESTA NO ES LA VERSIÓN BUENA DEL AUTOENCODER, SOLO PRIMER SKETCH
############################################

'''

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


print("Hello")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device


csv_dir= "/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv"


 
def extrac_neg(csv_dir):
    metadata= pd.read_csv(csv_dir)
    negatives = metadata[metadata['DENSITAT'] == 'NEGATIVA']
    image_names = negatives['CODI'].tolist()
    image_names = [name + "_1" for name in image_names]

    return image_names

#
image_names=extrac_neg(csv_dir)
len(image_names)

#
train=image_names[0:16]

#
data_dir= "/fhome/mapsiv/QuironHelico/CroppedPatches/"

#
def extract_patch_paths(data_dir,lpaths):
    selected_images = []

    for image_name in lpaths:
        folder_path = os.path.join(data_dir, image_name)
        if os.path.exists(folder_path):
            # Obtén todos los archivos (imágenes) dentro de la carpeta
            images_in_folder = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if image.endswith(".png")]
            selected_images.extend(images_in_folder)
    return selected_images

#
selected_patches= extract_patch_paths(data_dir, train)
len(selected_patches)

#
class CustomDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        image = Image.open(patch_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

#
transform = transforms.Compose([transforms.Resize((253, 253)), transforms.ToTensor()])

#
train_dataset = CustomDataset(selected_patches, transform)
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#
for batch in train_dataloader:
    for image in batch:
      img=image
    print(batch.shape)
    break


#
img=img.permute(1,2,0)

plt.imshow(img)
plt.axis('off')  # Desactiva los ejes
plt.show()

#
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Bloque 1: Conv1 -> BatchNorm1 -> LeakyReLU1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Puedes usar Sigmoid para la reconstrucción si tus imágenes están en el rango [0, 1]
        )

        # Bloque 2: Conv2 -> BatchNorm2 -> LeakyReLU2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Bloque 3: Conv3 -> BatchNorm3 -> LeakyReLU3
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Codificación
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        # Decodificación
        y2 = self.decoder3(x3)
        y1 = self.decoder2(y2)
        y = self.decoder1(y1)

        return y



#
autoencoder = Autoencoder()

#
autoencoder= autoencoder.to(device)

#
criterion = nn.MSELoss()

# Define el optimizador (por ejemplo, SGD)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Número de épocas
num_epochs = 2

# Entrenamiento
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_dataloader:
        # Obtén los datos de entrada
        inputs = batch

        # Pasa los datos al dispositivo (por ejemplo, GPU)
        inputs = inputs.to(device)

        # Reinicia los gradientes
        optimizer.zero_grad()

        # Propagación hacia adelante
        outputs = autoencoder(inputs)

        # Calcula la pérdida de reconstrucción
        loss = criterion(outputs, inputs)

        # Retropropagación y optimización
        loss.backward()
        optimizer.step()

        # Estadísticas de pérdida
        running_loss += loss.item()

    # Imprime la pérdida promedio en cada época
    print(f'Época {epoch + 1}, Pérdida: {running_loss / len(train_dataloader)}')

print('Entrenamiento completado')


#



