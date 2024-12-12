import numpy as np
from scipy.ndimage import convolve
from imageio import imread
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = imread("C:\Daun singkong.jpeg")
gray_image = rgb2gray(image)

roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])
edges_roberts_x = convolve(gray_image, roberts_x)
edges_roberts_y = convolve(gray_image, roberts_y)
edges_roberts = np.sqrt(edges_roberts_x**2 + edges_roberts_y**2)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
edges_sobel_x = convolve(gray_image, sobel_x)
edges_sobel_y = convolve(gray_image, sobel_y)
edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)

plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges_roberts, cmap='gray')
plt.title("Roberts Edge Detection")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edges_sobel, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.tight_layout()
plt.show()
