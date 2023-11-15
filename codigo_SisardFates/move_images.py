import shutil
import csv
import matplotlib.pyplot as plt
import os

source_directory = r'C:\Users\SFATESS\Documents\Sisard\GPU\images'
destination_directory = r'C:\Users\SFATESS\Documents\Sisard\GPU\clean_images'

def move_image(source_directory, destination_directory):
    for foldername in os.listdir(source_directory):
        folder_path = os.path.join(source_directory, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, os.path.join(destination_directory, filename))

    print("All image files have been copied to the destination directory.")

def remove_non_png_files(directory_path):
    file_list = os.listdir(directory_path)
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and not filename.lower().endswith('.png'):
            os.remove(file_path)
            print(f"Removed: {filename}")
    
    print("Files not ending with '.png' have been removed.")

def plot_losses_from_csv(csv_file_path):
    iterations = []
    generator_losses = []
    discriminator_losses = []

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iterations.append(int(row['Iteration']))
            generator_losses.append(float(row['Generator Loss']))
            discriminator_losses.append(float(row['Discriminator Loss']))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, generator_losses, label='Generator Loss', color='b')
    plt.plot(iterations, discriminator_losses, label='Discriminator Loss', color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.show()

def remove_all_images(directory_path = r'C:\Users\SFATESS\Documents\Sisard\GPU\generated_images'):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp']
    file_list = os.listdir(directory_path)
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            os.remove(file_path)
            print(f"Removed image: {filename}")
    
    print("All image files have been removed.")
