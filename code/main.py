from scipy.signal import convolve2d
import numpy as np
import skimage.io as skio
from utils import save_image, gaussian_blur, gradient_edge, gradient_edge_fast, sharpen, sharpen_fast

# ======== PART 1: Filters ========

# ==== 1.1: Finite Difference Operator ====
# Generate edge image for camera man using binarized gradient magnitude
cameraman_image = skio.imread(f'../data/cameraman.png', as_gray=True)
edge_image = gradient_edge(cameraman_image)
save_image(edge_image, 'cameraman_edges:gradient')

# ==== 1.2: Derivative of Gaussian Filter ====
# Generate less noisy edge image using gaussian blur + binarized gradient magnitude
blurred_image = gaussian_blur(cameraman_image, 5, 1)
save_image(blurred_image, 'cameraman_blurred:gaussian')
less_noisy_edge_image = gradient_edge(blurred_image, 0.125)
save_image(less_noisy_edge_image, 'cameraman_edges:blur-gradient')

# Combine gaussian blur and derivative convolutions into a single convolution (optimization)
less_noisy_edge_image = gradient_edge_fast(cameraman_image, threshold=0.125)
save_image(less_noisy_edge_image, 'cameraman_edges:dog')

# ======== PART 2: Frequencies ========
# ==== 2.1: Image Sharpening ====
# Sharpen the Taj Mahal image using unsharp masking
taj_image = skio.imread(f'../data/taj.jpg')
sharpened_image = sharpen(taj_image)
save_image(sharpened_image, 'taj_sharpened:unsharp_mask')

# Do the same by convolving the image with the unsharp mask filter (optimization)
sharpened_image = sharpen_fast(taj_image)
save_image(sharpened_image, 'taj_sharpened:unsharp_mask_filter')
