from scipy.signal import convolve2d
import numpy as np
import skimage.io as skio
from utils import save_grayscale_image

# ======== PART 1: Filters ========
# 1.1: Finite Difference Operator

# X and Y derivative images
cameraman_im = skio.imread(f'../data/cameraman.png', as_gray=True)
D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])
x_derivative = convolve2d(cameraman_im, D_x, mode='same')
y_derivative = convolve2d(cameraman_im, D_y, mode='same')
save_grayscale_image(x_derivative, 'cameraman_x_derivative')
save_grayscale_image(y_derivative, 'cameraman_y_derivative')

# Gradient magnitude image
gradient_magnitude = np.sqrt(x_derivative**2 + y_derivative**2)
save_grayscale_image(gradient_magnitude, 'cameraman_gradient_magnitude')

# Binarize the gradient magnitude image
GRADIENT_THRESHOLD = 0.3
edge_image = gradient_magnitude >= GRADIENT_THRESHOLD
save_grayscale_image(edge_image, 'cameraman_edge_image')
