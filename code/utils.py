import numpy as np
import skimage.io as skio
import cv2
from scipy.signal import convolve2d
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage import color

# ========== Parameters ============
DEFAULT_GAUSSIAN_KSIZE = 5
DEFAULT_GAUSSIAN_SD = 1
DEFAULT_GRADIENT_THRESHOLD = 0.3

# ========== Helper functions ============

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

# Return a hybrid of two aligned images
def hybrid(im1, im2, cutoff_ksize, cutoff_sigma):
    gray_im1, gray_im2 = color.rgb2gray(im1), color.rgb2gray(im2)
    high_freq_im1 = gray_im1 - gaussian_blur(gray_im1, cutoff_ksize, cutoff_sigma)
    low_freq_im2 = gaussian_blur(gray_im2, cutoff_ksize, cutoff_sigma)
    return np.clip(low_freq_im2 + high_freq_im1, 0, 1)

# ========== Starter code functions ============
def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, (dscale, dscale, 1)) # should be same as passing multichannel=True
    else:
        im2 = sktr.rescale(im2, (1./dscale, 1./dscale, 1))
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2