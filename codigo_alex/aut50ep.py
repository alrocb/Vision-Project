
 
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
from tqdm import tqdm

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
device

 
csv_dir= "/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv"


 
def extrac_neg(csv_dir):
    metadata= pd.read_csv(csv_dir)
    negatives = metadata[metadata['DENSITAT'] == 'NEGATIVA']
    image_names = negatives['CODI'].tolist()
    image_names = [name + "_1" for name in image_names]

    return image_names

 
image_names=extrac_neg(csv_dir)
len(image_names)

 
train=image_names[0:16]

 
test= image_names[16:18]

 
data_dir= "/fhome/mapsiv/QuironHelico/CroppedPatches/"

 
def extract_patch_paths(data_dir,lpaths):
    selected_images = []

    for image_name in lpaths:
        folder_path = os.path.join(data_dir, image_name)
        if os.path.exists(folder_path):
            # Obtén todos los archivos (imágenes) dentro de la carpeta
            images_in_folder = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if image.endswith(".png")]
            selected_images.extend(images_in_folder)
    return selected_images

 
train_patches= extract_patch_paths(data_dir, train)
len(train_patches)

 
test_patches= extract_patch_paths(data_dir, test)
len(test_patches)

 
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

 
transform = transforms.Compose([transforms.Resize((253, 253)), transforms.ToTensor()])

 
train_dataset = CustomDataset(train_patches, transform)
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

 
test_dataset = CustomDataset(test_patches, transform)
batch_size = 128
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

 
for batch in train_dataloader:
    for image in batch:
      img=image
    print(batch.shape)
    break


 
img=img.permute(1,2,0)

plt.imshow(img)
plt.axis('off')  # Desactiva los ejes
plt.show()

 
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



 
autoencoder = Autoencoder()

 
autoencoder= autoencoder.to(device)

 
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define el modelo Autoencoder aquí
# ...

# Define el dataloader de entrenamiento (train_dataloader) y prueba (test_dataloader) aquí
# ...

criterion = nn.MSELoss()

# Define el optimizador (por ejemplo, Adam)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Número de épocas
num_epochs = 50

# Path para guardar el modelo
model_path = "/fhome/gia06/codigo/models/model1_prueba.pt"

# Listas para almacenar la pérdida de entrenamiento y prueba en cada época
train_loss_history = []
test_loss_history = []

# Entrenamiento
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        inputs = batch
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calcula la pérdida promedio de entrenamiento en esta época
    train_loss = running_loss / len(train_dataloader)
    train_loss_history.append(train_loss)

    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Test'):
            inputs = batch
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)
            test_loss += criterion(outputs, inputs).item()

    # Calcula la pérdida promedio en el conjunto de prueba en esta época
    test_loss = test_loss / len(test_dataloader)
    test_loss_history.append(test_loss)

    # Guarda el modelo
    torch.save(autoencoder.state_dict(), model_path)

    # Imprime la pérdida promedio en cada época
    print(f'Época {epoch + 1}, Pérdida de Entrenamiento: {train_loss}, Pérdida de Prueba: {test_loss}')

# Plotea la curva de pérdida en el conjunto de entrenamiento y prueba
plt.figure()
plt.plot(train_loss_history, label='Entrenamiento')
plt.plot(test_loss_history, label='Prueba')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Curva de Pérdida')
plt.show()

# Muestra algunas imágenes y sus reconstrucciones después de cada época
with torch.no_grad():
    original_images = []
    reconstructed_images = []

    for i, batch in enumerate(test_dataloader):
        if i >= 3:  # Muestra solo 3 imágenes
            break

        inputs = batch
        inputs = inputs.to(device)
        outputs = autoencoder(inputs)

        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        index = np.random.randint(0, len(inputs))

        original_images.append(inputs[index])
        reconstructed_images.append(outputs[index])

    for i in range(3):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(original_images[i], (1, 2, 0)))
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(reconstructed_images[i], (1, 2, 0)))
        plt.title('Reconstrucción')

        plt.show()



