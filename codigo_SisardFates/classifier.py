from model_def import *
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import pandas as pd

def load_discriminator(model_path):
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(model_path))
    return discriminator

def classify_images_in_directory(image_folder, model_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CustomDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_discriminator(model_path).to(device)
    model.eval()

    probabilities_dict = {}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            outputs = model(batch)
            probabilities = outputs.cpu().numpy()

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))
            filenames = dataset.image_paths[start_idx:end_idx]

            probabilities_dict.update({os.path.basename(os.path.dirname(filename)) + os.path.splitext(os.path.basename(filename))[0]: probability[0] for filename, probability in zip(filenames, probabilities)})

    return probabilities_dict

import os

def classify_images_in_main_directory(main_directory, model_path, batch_size=32):
    all_predictions = {}

    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)

        if os.path.isdir(folder_path):
            print(f"Classifying images in folder: {folder_name}")
            folder_predictions = classify_images_in_directory(folder_path, model_path, batch_size)
            all_predictions.update({f"{folder_name}.{key}": value for key, value in folder_predictions.items()})

    return all_predictions

def add_discriminator_column(csv_file_path, id_column_name, discriminator_dict, output_csv_path):
    df = pd.read_csv(csv_file_path)
    df['discriminator'] = df[id_column_name].astype(str).map(discriminator_dict)
    df.to_csv(output_csv_path, index=False)

main_directory = r'C:\Users\SFATESS\Documents\Sisard\GPU\AnnotatedPatches'
model_path = 'discriminator.pth'
all_predictions = classify_images_in_main_directory(main_directory, model_path)


csv_file_path = r'C:\Users\SFATESS\Documents\Sisard\GPU\AnnotatedPatches\window_metadata.csv'
id_column_name = 'ID'
output_csv_path = 'predictions.csv'

add_discriminator_column(csv_file_path, id_column_name, all_predictions, output_csv_path)
