import os
import imageio
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = imageio.imread(img_path) / 255.0
            images.append(img)
    return images

def trim_images(images, ax, ay, bx, by):
    return np.array([img[ax:bx, ay:by] for img in images])