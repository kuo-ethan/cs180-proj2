import numpy as np
import skimage.io as skio
import cv2
from scipy.signal import convolve2d

# PARAMETERS AND CONSTANTS
DEFAULT_GAUSSIAN_KSIZE = 5
DEFAULT_GAUSSIAN_SD = 1
DEFAULT_GRADIENT_THRESHOLD = 0.3

# Normalize and save images, automatically accounting for grayscale or RGB
def save_image(image, filename):
    if len(image.shape) == 2:  # Grayscale image
        if image.dtype == bool:
            image_uint8 = image.astype(np.uint8) * 255
        else:
            image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
            image_uint8 = (image_normalized * 255).astype(np.uint8)
    elif len(image.shape) == 3:  # RGB image
        image_uint8 = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):
            channel = image[:, :, i]
            channel_normalized = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
            image_uint8[:, :, i] = (channel_normalized * 255).astype(np.uint8)
    skio.imsave(f'../images/{filename}.jpg', image_uint8)

# Apply a gaussian low pass filter over an image
def gaussian_blur(image, ksize=5, sigma=1):
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d.T)

    if len(image.shape) == 2:  # Grayscale image
        blurred_image = convolve2d(image, gaussian_2d, mode='same')
    
    elif len(image.shape) == 3:  # RGB image
        blurred_image = []
        for i in range(3):
            blurred_image.append(convolve2d(image[:, :, i], gaussian_2d, mode='same'))
        blurred_image = np.dstack([channel for channel in blurred_image])
    
    return blurred_image

# Compute the gradient magnitude of the image, the binarize it to accentuate edges
def gradient_edge(image, threshold=DEFAULT_GRADIENT_THRESHOLD):
    # X and Y partial derivative images
    D_x = np.array([[1, -1]])
    D_y = np.array([[1], [-1]])
    x_derivative = convolve2d(image, D_x, mode='same')
    y_derivative = convolve2d(image, D_y, mode='same')

    # Gradient magnitude image
    gradient_magnitude = np.sqrt(x_derivative**2 + y_derivative**2)

    # Binarize the gradient magnitude image
    edge_image = gradient_magnitude >= threshold
    return edge_image

# Optimized version of gaussian_blur + gradient_edge that uses a single convolution with
# derivative of gaussian filters
def gradient_edge_fast(image, ksize=DEFAULT_GAUSSIAN_KSIZE, sigma=DEFAULT_GAUSSIAN_SD, threshold=DEFAULT_GRADIENT_THRESHOLD):
    # X and Y gaussian derivatives
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d.T)
    D_x = np.array([[1, -1]])
    D_y = np.array([[1], [-1]])
    DoG_x = convolve2d(gaussian_2d, D_x)
    DoG_y = convolve2d(gaussian_2d, D_y)
    x_gaussian_derivative = convolve2d(image, DoG_x, mode='same')
    y_gaussian_derivative = convolve2d(image, DoG_y, mode='same')

    # Gradient magnitude image
    gradient_magnitude = np.sqrt(x_gaussian_derivative**2 + y_gaussian_derivative**2)

    # Binarize the gradient magnitude image
    edge_image = gradient_magnitude >= threshold
    return edge_image

# Sharpen the image using unsharp mask technique
def sharpen(image, ksize=DEFAULT_GAUSSIAN_KSIZE, sigma=DEFAULT_GAUSSIAN_SD, alpha=2.0):
    low_freq_image = gaussian_blur(image, ksize, sigma)
    high_freq_image = image - low_freq_image
    sharpened_image = image + alpha * high_freq_image
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

# Sharpen the image by convolving with the unsharp mask filter
def sharpen_fast(image, ksize=DEFAULT_GAUSSIAN_KSIZE, sigma=DEFAULT_GAUSSIAN_SD, alpha=2.0):
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d.T)
    identity_filter = create_identity_filter(ksize)
    unsharp_mask_filter = (1 + alpha) * identity_filter - alpha * gaussian_2d

    # Apply the filter to each color channel
    sharpened_image = []
    for i in range(3):
        sharpened_image.append(convolve2d(image[:, :, i], unsharp_mask_filter, mode='same'))
    sharpened_image = np.dstack([channel for channel in sharpened_image])
    return np.clip(sharpened_image, 0, 255).astype(np.uint8) 

# Function to create the identity filter. Size should be odd.
def create_identity_filter(size):
    identity_filter = np.zeros((size, size))
    center = size // 2
    identity_filter[center, center] = 1
    return identity_filter