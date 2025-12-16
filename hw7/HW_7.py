# gabor filter combined feature
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read grayscale image
img = cv2.imread("leaf.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread("leaf.png")

# Step 2: Define Gabor filter parameters
ksize = 51
sigmas = [2.0, 4.0]
lambd = 10.0
gamma = 0.5
psi = 0
orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Step 3: Apply Gabor filters and normalize responses
# --------------------- TBD ---------------------------------
# 3.1 Iterate (nested) over all values of sigma and orientation.
# 3.2 For each combination, compute the Gabor kernel using cv2.getGaborKernel() with the given filter parameters.
# 3.3 Apply the kernel to the image using cv2.filter2D().
# 3.4 Normalize the filtered output to the range [0, 255], then convert it to uint8 type for visualization.
# 3.5 Append the normalized output to a list of filter responses.
# 3.6 Pass the list of appended outputs to Step 4 for further processing.

responses = []
for sigma in sigmas:
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)        
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)        
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)        
        responses.append(normalized)

# Step 4: Combine all responses (max response at each pixel)
combined = np.maximum.reduce(responses)
# -------------------- TBD - END ---------------------------------

# step 5: visualize
heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
highlighted = cv2.addWeighted(img_color, 0.6, heatmap, 0.4, 0)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(combined, cmap='gray')
plt.title("Combined Gabor Response")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
plt.title("Reprojected Texture Map")
plt.axis('off')

plt.tight_layout()
plt.show()
