import numpy as np
import cv2
import bm3d

def normalize_image(image_path, use_bm3d=True):
    """
    Normalize a single input image and optionally apply BM3D filtering.

    Args:
    image_path (str): Path to the input image file.
    use_bm3d (bool): Whether to apply BM3D filtering after normalization.

    Returns:
    np.array: Normalized (and optionally filtered) image.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Apply BM3D filtering if requested
    if use_bm3d:
        normalized_image = bm3d.bm3d(image, sigma_psd=2/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    # Min-max normalization
    min_val = np.min(normalized_image)
    max_val = np.max(normalized_image)
    normalized_image = (normalized_image - min_val) / (max_val - min_val)

    return normalized_image

def print_image_stats(image):
    """
    Print basic statistics of the image.
    """
    print(f"Image statistics:")
    print(f"  Mean: {np.mean(image):.4f}")
    print(f"  Std Dev: {np.std(image):.4f}")
    print(f"  Min: {np.min(image):.4f}")
    print(f"  Max: {np.max(image):.4f}")
    print(f"  Shape: {image.shape}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize an input image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--no-bm3d", action="store_true", help="Disable BM3D filtering")
    args = parser.parse_args()

    # Normalize the image
    normalized_image = normalize_image(args.image_path, use_bm3d=not args.no_bm3d)

    # Print statistics
    print_image_stats(normalized_image)

    # Optionally, you can save the normalized image
    cv2.imwrite("normalized_image.png", (normalized_image * 255).astype(np.uint8))