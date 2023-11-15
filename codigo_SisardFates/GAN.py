import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import torch.optim as optim
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_paths = os.listdir(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_folder = r'C:\Users\SFATESS\Documents\Sisard\GPU\clean_images'
dataset = CustomDataset(data_folder, transform=transform)

batch_size = 120
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(dim=2).squeeze(dim=2)
        return x

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator.to(device)
discriminator.to(device)
criterion.to(device)

generator_losses = []
discriminator_losses = []

num_epochs = 20
save_interval = 5
output_folder = 'generated_images'
os.makedirs(output_folder, exist_ok=True)

for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, 100, 1, 1).to(device)
        fake_images = generator(noise)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        optimizer_D.zero_grad()
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)
        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        outputs_fake = discriminator(fake_images)
        g_loss = criterion(outputs_fake, real_labels)
        g_loss.backward()
        optimizer_G.step()

        generator_losses.append(g_loss.item())
        discriminator_losses.append(d_loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')



    if (epoch + 1) % save_interval == 0:
        with torch.no_grad():
            fake_samples = generator(torch.randn(2, 100, 1, 1).to(device))
        for idx, fake_sample in enumerate(fake_samples):
            fake_image = transforms.ToPILImage()(0.5 * fake_sample + 0.5)
            fake_image.save(os.path.join(output_folder, f'generated_image_epoch{epoch + 1}_sample{idx}.png'))




with open('gan_losses.csv', 'w', newline='') as csvfile:
    fieldnames = ['Iteration', 'Generator Loss', 'Discriminator Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for iteration, (gen_loss, disc_loss) in enumerate(zip(generator_losses, discriminator_losses)):
        writer.writerow({'Iteration': iteration, 'Generator Loss': gen_loss, 'Discriminator Loss': disc_loss})
