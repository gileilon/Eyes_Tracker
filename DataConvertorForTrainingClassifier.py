import os
import numpy as np
from PIL import Image
from tqdm import tqdm

class DataConvertor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def convert_images_to_pixels(self):
        pixels_vectors = []
        file_names = os.listdir(self.folder_path)

        for file_name in tqdm(file_names, desc='loading files'):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.folder_path, file_name)
                try:
                    image = Image.open(image_path).convert('L')
                    flattened_vector = np.array(image).flatten()
                    pixels_vectors.append(flattened_vector)
                    image.close()
                except Exception as e:
                    print(f"Error processing image {file_name}: {e}")

        pixels_vectors_for_classifier = np.array(pixels_vectors)
        return pixels_vectors_for_classifier


