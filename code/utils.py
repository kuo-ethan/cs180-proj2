import numpy as np
import skimage.io as skio

# Normalize and save images as grayscale JPG
def save_grayscale_image(image, filename):
    if image.dtype == bool:
        image_uint8 = image.astype(np.uint8) * 255
    else:
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        image_uint8 = (image_normalized * 255).astype(np.uint8)
    skio.imsave(f'../images/{filename}.jpg', image_uint8)