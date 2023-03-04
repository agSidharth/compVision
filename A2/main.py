import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def hessian_corner_detector(image, sigma=1, threshold=0):
    # Calculate the Hessian matrix
    Hxx = ndimage.gaussian_filter(image, sigma=(sigma, sigma), order=(0, 2))
    Hyy = ndimage.gaussian_filter(image, sigma=(sigma, sigma), order=(2, 0))
    Hxy = ndimage.gaussian_filter(image, sigma=(sigma, sigma), order=(1, 1))
    # Calculate the determinant and trace of the Hessian matrix
    detH = (Hxx * Hyy) - (Hxy ** 2)
    traceH = Hxx + Hyy
    # Calculate the corner response function
    R = detH - 0.06 * (traceH ** 2)
    # Threshold the corner response function
    R[R < threshold] = 0
    # Non-maximum suppression
    R = ndimage.maximum_filter(R, size=(3, 3))
    R[R < threshold] = 0
    R[R > 0] = 1
    return R

 # Load an image
image = plt.imread('lena.png')

# Convert to grayscale
image = np.mean(image, axis=2)

# Apply the Hessian corner detector
corners = hessian_corner_detector(image, sigma=2, threshold=0.01)

# Display the results
fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.imshow(image, cmap='gray')
ax0.set_title('Original Image')
ax1.imshow(corners, cmap='gray')
ax1.set_title('Detected Corners')
plt.show()
