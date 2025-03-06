"""
============
Binary Image
============

Previously, we learned that defining a threshold value
can help identify the signal in an image. The result of this can be represented by a binary image.

You can think of a binary image as a map showing where the objects of interest are located.
Pixels belonging to the signal become ``True`` (or 1), and pixels belonging to the background become ``False`` (or 0).

Let's create a binary image from our example `blobs.jpeg` by applying Otsu's thresholding method:
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu

blobs = Image.open("images/blobs.jpeg")
blobs = np.array(blobs)

th_otsu = threshold_otsu(blobs)
bin_image = blobs > th_otsu

plt.figure()

plt.subplot(121)
plt.title('Image')
plt.imshow(blobs, cmap='gray')

plt.subplot(122)
plt.title('Binary image (mask)')
plt.imshow(bin_image, cmap='gray')

plt.tight_layout()
plt.show()

######################################################################
# It's called "binary" because it contains only two values — (1 and 0)
# or (``True`` and ``False``) — corresponding to the pixels we want
# (shown in white) and the pixels that we don't want (shown in black),
# respectively.
#

print(f'unique values of original image: {np.unique(blobs)}')
print(f'unique values of binary image: {np.unique(bin_image)}')

######################################################################
# The ``bin_image`` image above maps the location of signals, which is defined
# by pixels with a value greater than the threshold ``blobs > th_otsu``.
#
# But we can make a binary image of any condition(s) that we want.
# For example, we can make binary images of:
#
# - Background, i.e. pixels that are lower than the threshold with ``image < th``.
# - Pixels with maximum intensity with ``image == max(image)``
# - Pixels with values in certain range (value 1 < image < value 2), can be defined as ``(image > val_1) & (image < val_2)``
#

# pixels lower than threshold
background = blobs < th_otsu

# pixels with maximum intensity
max_value = blobs == blobs.max()

# pixels with values in ceratin range
edges = (blobs > 50) & (blobs < 200)

plt.figure(figsize=(12,4.5))

plt.subplot(131)
plt.title("Background (image < th)")
plt.imshow(background, cmap='gray')

plt.subplot(132)
plt.title("Max intensity (image == max(image))")
plt.imshow(max_value, cmap='gray')

plt.subplot(133)
plt.title("Edges (50 < image < 200)")
plt.imshow(edges, cmap='gray')

plt.tight_layout()
plt.show()