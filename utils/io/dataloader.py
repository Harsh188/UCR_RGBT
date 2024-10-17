from PIL import Image
import numpy as np
import os

class ImageDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.images = []

    def load_images(self):
        """
        Load all PNG images from the specified directory structure.
        """
        for root, _, files in os.walk(self.base_path):
            if self._is_valid_directory(root):
                self._process_files(root, files)
        return np.array(self.images)

    def _is_valid_directory(self, root):
        """
        Check if the current directory is a valid LWIR data directory.
        """
        return all(subdir in root for subdir in ['lwir', 'raw', 'data'])

    def _process_files(self, root, files):
        """
        Process PNG files in the current directory.
        """
        for file in files:
            if file.lower().endswith('.png'):
                self._load_and_append_image(os.path.join(root, file))

    def _load_and_append_image(self, img_path):
        """
        Load an image from the given path and append it to the images list.
        """
        with Image.open(img_path) as img:
            img_array = np.array(img)
            self.images.append(img_array)

    def get_images(self):
        """
        Return the loaded images.
        """
        return self.images

    def __repr__(self):
        return f"ImageDataLoader(base_path='{self.base_path}', num_images={len(self.images)})"