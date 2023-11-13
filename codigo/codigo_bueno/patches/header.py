import cv2
import SimpleITK as sitk
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

def encontrar_contornos(canal,stack_resized):

    contornos, _ = cv2.findContours(canal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    regiones_interes = []
    for contorno in contornos:
        print("Contorno detectado")
        x, y, w, h = cv2.boundingRect(contorno)
        print(x,y,w,h)

        region = stack_resized[y:y+h, x:x+w]
        regiones_interes.append(region)
    return regiones_interes,contornos

def sliding_window(imagen_bin,imagen, window_size):
    valid_patches = []
    for y in range(0, imagen_bin.shape[0] - window_size[0], 7 ):
        for x in range(0, imagen_bin.shape[1] - window_size[1], 7 ):
            # Extrae el patch actual
            patch = imagen_bin[y:y+window_size[0], x:x+window_size[1]]

            white_pixels = cv2.countNonZero(patch)
            black_pixels = patch.size - white_pixels

            if white_pixels > 0 and black_pixels > 0:
                patch_imagen_original = imagen[y:y+window_size[0], x:x+window_size[1]]
                valid_patches.append(patch_imagen_original)
    return valid_patches


def extract_patches(rute):
    tif_stack = tf.imread(rute)
    print("The shape of your image is:",tif_stack.shape)
    print("Resizing image...")
    alto,ancho,canales= tif_stack.shape
    antiguo_aspect_ratio= ancho/alto

    nuevo_ancho = int(ancho/100)  
    nuevo_alto = int(alto/100)  

    nuevo_aspect_ratio= nuevo_ancho/ nuevo_alto


    stack_resized = cv2.resize(tif_stack, (nuevo_ancho, nuevo_alto))
    
    print("The new shape is:",stack_resized.shape)
     
    print("Practically we preserve all the aspect ratio when resizing since the original was:",antiguo_aspect_ratio,"and the new one is:", nuevo_aspect_ratio)
     
    print("Plotting image...")
     
    plt.imshow(stack_resized)
    plt.show()
    
    print("Converting image from BGR to HSV...")
    imagen_hsv = cv2.cvtColor(stack_resized, cv2.COLOR_BGR2HSV)
     
    print("Extracting Hue channel...")
    canal= imagen_hsv[:,:,:1]
     
    print("Canal extracted! Shape:",canal.shape)
     
    print("Visualizing..")
    plt.imshow(canal)
    plt.show()
    print("Preparing to extract contours..")
    regiones_interes,contornos=encontrar_contornos(canal,stack_resized)

    imagen_con_contornos = stack_resized.copy()

    print("Visualizing extracted contours..")
    # Dibuja los contornos en la imagen
    cv2.drawContours(imagen_con_contornos, contornos, -1, (255,0, 0), 9)
    plt.imshow(imagen_con_contornos)
    plt.axis('off')  # Desactiva los ejes
    plt.show()

     
    for i in range(len(regiones_interes)):
        crop= regiones_interes[i]
         
        print("Printing crop number:",i)
        plt.imshow(crop)
        plt.axis('off') 
        plt.show()
         

    print("Time to process the cropped images to extract patches..")

    for i in range(len(regiones_interes)):
        print("Extracting saturation channel from",i,"crop...")
        imagen_hsv = cv2.cvtColor(regiones_interes[i], cv2.COLOR_BGR2HSV)
        canal_crop= imagen_hsv[:,:,1]
         
        plt.imshow(canal_crop, cmap="gray")
        plt.axis('off') 
        plt.show()

         
        print("Binarizing image...")
        umbral, imagen_binaria = cv2.threshold(canal_crop, 60, 120, cv2.THRESH_BINARY)
         
        plt.imshow(imagen_binaria, cmap="gray")
        plt.axis('off') 
        plt.show()

         

        print("Applying morphological operation to avoid holes and noise... (opening)")
 
        kernel = np.ones((3, 3), np.uint8)  

        imagen_abierta = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)
         

        plt.imshow(imagen_abierta, cmap="gray")
        plt.axis('off') 
        plt.show()
         

        print("Defining window for sliding window process..")
        window_size=(15,15)
        patches=sliding_window(imagen_abierta,regiones_interes[i],window_size)
         
        print("Got it! Patches obtained:",len(patches))
         

        print("Visualizing 5 first patches..")
        for i in range (0,5):
            plt.imshow(patches[i])
            plt.axis('off')  
            plt.show()


          
    print("Patches extraction process finished successfully!")
    #return patches
