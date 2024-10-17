'''
Normalizes input image.
'''

import numpy as np
import bm3d
import cv2
import logging

class ImageNormalizer:
    def __init__(self, ImageDataLoader, log_level=logging.INFO):
        self.data_loader = ImageDataLoader
        # Load images using ImageDataLoader.
        self.images = self.data_loader.load_images()

        self.normalized_images = None

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def normalize(self):
        """
        Normalize images using the specified method.

        Args:
        method (str): Normalization method ('minmax' or 'zscore').

        Returns:
        np.array: Normalized image data.
        """
        if self.images is None:
            raise ValueError("Images not loaded. Please load images first.")
        
        self.logger.info(self.get_dataset_stats())

        # Normalize
        imgs_norm = self._min_max_normalization()

        # BM3D filtering
        imgs_norm = self._bm3d_filtering(imgs_norm)

        return imgs_norm

    def _bm3d_filtering(self, imgs):
        imgs_norm = []
        for img in imgs:
            img_norm = bm3d.bm3d(img, sigma_psd=2/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
            imgs_norm.append(img_norm)
        imgs_norm = np.array(imgs_norm)
        return imgs_norm

    def _min_max_normalization(self):
        """
        Perform min-max normalization on the image data.
        """
        min_val = np.min(self.images)
        max_val = np.max(self.images)
        return (self.images - min_val) / (max_val - min_val)

    def _z_score_normalization(self):
        """
        Perform z-score normalization on the image data.
        """
        mean = np.mean(self.images)
        std = np.std(self.images)
        return (self.images - mean) / std

    def get_dataset_stats(self):
        """
        Calculate and return basic statistics of the dataset.

        Returns:
        dict: Dictionary containing dataset statistics.
        """
        if self.images is None:
            raise ValueError("Images not loaded. Please load images first.")

        return {
            "mean": np.mean(self.images),
            "std": np.std(self.images),
            "min": np.min(self.images),
            "max": np.max(self.images),
            "shape": self.images.shape
        }

    def __repr__(self):
        return f"ImageNormalizer(base_path='{self.base_path}', num_images={len(self.images) if self.images is not None else 0})"

if __name__=="__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.io.dataloader import ImageDataLoader
    import pdb

    dl_obj = ImageDataLoader(base_path=r"C:\Users\EndUser\Documents\Programming\BosonChecks\UCR_RGBT\dataset\test")
    norm_obj = ImageNormalizer(dl_obj)
    norm_imgs = norm_obj.normalize()

    pdb.set_trace()
    