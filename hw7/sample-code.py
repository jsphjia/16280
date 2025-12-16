# this is just a sample code file for reference on the cv2 and matplotlib usage
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Read the image (ensure the file is in the same directory)
apple = cv2.imread('apple_rgb.png')

# Check if the image was loaded successfully
if apple is None:
    print("Error: Image not found!")
else:
    # Convert from BGR (OpenCV default) to RGB for proper display
    apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
    apple_r, apple_g, apple_b = cv2.split(apple_rgb)

    # Display the image using matplotlib
    plt.imshow(apple_rgb)
    plt.title('RGB Image')
    plt.axis('off')  # Hide axis ticks
    plt.show()

    # Display individual color channels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(apple_r, cmap='gray')
    axes[0].set_title('Red Channel')
    axes[0].axis('off')
    axes[1].imshow(apple_g, cmap='gray')
    axes[1].set_title('Green Channel')
    axes[1].axis('off')
    axes[2].imshow(apple_b, cmap='gray')
    axes[2].set_title('Blue Channel')
    axes[2].axis('off')
    plt.show()

    # Convert to grayscale
    apple_gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
    apple_hist = cv2.calcHist([apple_gray], [0], None, [256], [0, 256])

    # Plot the grayscale image and histogram together
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(apple_gray, cmap='gray')
    axes[0].set_title('Grayscale Image')
    axes[0].axis('off')
    axes[1].plot(apple_hist, color='black')
    axes[1].set_title('Grayscale Histogram')
    plt.show()
    
    # manual thresholding and Otsu's method
    _, binary_manual = cv2.threshold(apple_gray, 128, 255, cv2.THRESH_BINARY)
    _, binary_otsu = cv2.threshold(apple_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(apple_gray, cmap='gray')
    axes[0].set_title('Grayscale Image')
    axes[0].axis('off')
    axes[1].imshow(binary_manual, cmap='gray')
    axes[1].set_title('Manual Thresholding (128)')
    axes[1].axis('off')
    axes[2].imshow(binary_otsu, cmap='gray')
    axes[2].set_title("Otsu's Thresholding")
    axes[2].axis('off')
    plt.show()

    # sobel edge detection and canny edge detection
    sobel_x = cv2.Sobel(apple_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(apple_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    canny_edges = cv2.Canny(apple_gray, 100, 200)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(apple_gray, cmap='gray')
    axes[0].set_title('Grayscale Image')
    axes[0].axis('off')
    axes[1].imshow(sobel_magnitude, cmap='gray')
    axes[1].set_title('Sobel Edge Detection')
    axes[1].axis('off')
    axes[2].imshow(canny_edges, cmap='gray')
    axes[2].set_title('Canny Edge Detection')
    axes[2].axis('off')
    plt.show()

# transformations on flower.png
flower = cv2.imread('flower.png')
if flower is None:
    print("Error: Image not found!")
else:
    flower_rgb = cv2.cvtColor(flower, cv2.COLOR_BGR2RGB)
    horiztonal_flip = cv2.flip(flower_rgb, 1)
    vertical_flip = cv2.flip(flower_rgb, 0)
    ccw_rotation = cv2.rotate(flower_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    crop_size = 100
    h, w = flower_rgb.shape[:2]
    start_x = np.random.randint(0, w - crop_size)
    start_y = np.random.randint(0, h - crop_size)
    crop = flower_rgb[start_y:start_y + crop_size, start_x:start_x + crop_size]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(flower_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(horiztonal_flip)
    axes[1].set_title('Horizontal Flip')
    axes[1].axis('off')
    axes[2].imshow(vertical_flip)
    axes[2].set_title('Vertical Flip')
    axes[2].axis('off')
    axes[3].imshow(ccw_rotation)
    axes[3].set_title('90° CCW Rotation')
    axes[3].axis('off')
    axes[4].imshow(crop)
    axes[4].set_title('Random 100x100 Crop')
    axes[4].axis('off')
    plt.show()


