import numpy as np
import skimage.io as skio
from skimage import color
from utils import save_image, gaussian_blur, gradient_edge, \
gradient_edge_fast, sharpen, sharpen_fast, align_images, hybrid

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

# Sharpen my own images
image = skio.imread(f'../data/cookie.jpg')
sharpened_image = sharpen_fast(image, alpha=5)
save_image(sharpened_image, 'cookie_sharpened')

image = skio.imread(f'../data/pug.jpg')
sharpened_image = sharpen_fast(image, alpha=5)
save_image(sharpened_image, 'pug_sharpened')

image = skio.imread(f'../data/bros.jpg')
sharpened_image = sharpen_fast(image, alpha=15)
save_image(sharpened_image, 'bros_sharpened')

# Blur then sharpen an image
image = skio.imread(f'../data/me.jpg')
blurred_image = gaussian_blur(image, ksize=10, sigma=2)
save_image(blurred_image, 'me_blurred')
sharpened_image = sharpen_fast(blurred_image, alpha=10)
save_image(sharpened_image, 'me_blurred_then_sharpened')

# ==== 2.2: Hybrid Images ====
# Dereck and nutmeg
im1 = skio.imread(f'../data/nutmeg.jpg')
im2 = skio.imread(f'../data/derek.jpg')
im1, im2 = align_images(im1, im2)
hybrid_im = hybrid(im1, im2, 25, 16)
save_image(hybrid_im, 'derek_nutmeg_hybrid')

# Gojo and Levi (also show images in Fourier domain)
im1 = skio.imread(f'../data/levi.jpg')[:,:,:3]
im2 = skio.imread(f'../data/gojo.jpg')[:,:,:3]
im1, im2 = align_images(im1, im2)
gray_im1, gray_im2 = color.rgb2gray(im1), color.rgb2gray(im2)

cutoff_freq = (25, 16)
high_freq_im1 = gray_im1 - gaussian_blur(gray_im1, cutoff_freq[0], cutoff_freq[1])
low_freq_im2 = gaussian_blur(gray_im2, cutoff_freq[0], cutoff_freq[1])
hybrid_im = np.clip(low_freq_im2 + high_freq_im1, 0, 1)
save_image(hybrid_im, 'levi_gojo_hybrid')

save_image(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_im1)))), 'levi_fourier')
save_image(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_im2)))), 'gojo_fourier')
save_image(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_freq_im1)))), 'levi_high_freq_fourier')
save_image(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_freq_im2)))), 'gojo_low_freq_fourier')
save_image(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_im)))), 'levi_gojo_hybrid_fourier')

# Young and old Ethan
im1 = skio.imread(f'../data/young.PNG')[:,:,:3]
im2 = skio.imread(f'../data/old.PNG')[:,:,:3]
im1, im2 = align_images(im1, im2)
hybrid_im = hybrid(im1, im2, 40, 30)
save_image(hybrid_im, 'ethan_hybrid')

# MJ and Kobe fadeaways (Kobe colored)
im1 = skio.imread(f'../data/kobe.png')[:,:,:3]
im2 = skio.imread(f'../data/mj.png')[:,:,:3]
im1, im2 = align_images(im1, im2)

cutoff_freq = (15, 10)
color_im1, gray_im2 = im1, color.rgb2gray(im2)
low_freq_im2 = gaussian_blur(gray_im2, cutoff_freq[0], cutoff_freq[1])
low_freq_im2 = np.dstack([low_freq_im2] * 3)  # Convert grayscale to 3-channel
low_freq_im1 = gaussian_blur(color_im1, cutoff_freq[0], cutoff_freq[1])
high_freq_im1 = color_im1 - low_freq_im1
hybrid_im = np.clip(low_freq_im2 + high_freq_im1, 0, 1)
save_image(hybrid_im, 'kobe_mj_hybrid')

# ==== 2.2: Gaussian and Laplacian Stacks ====
