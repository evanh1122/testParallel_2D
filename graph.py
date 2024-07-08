import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os

# NOTE - this currently plots all of grid1 and all of grid2

# this is the 2D array that we will fill with the Grid data
arrayCopy = np.loadtxt("dataCopy.txt")
arrayOriginal = np.loadtxt("dataOriginal.txt")


# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Plot arrayCopy in the first subplot using a gradient colormap
img1 = axs[0].imshow(arrayCopy, interpolation='nearest', cmap='plasma')
axs[0].set_title('Array Copy')

# Plot arrayOriginal in the second subplot using the same gradient colormap
img2 = axs[1].imshow(arrayOriginal, interpolation='nearest', cmap='plasma')
axs[1].set_title('Array Original')

# Make a color bar for the images
fig.colorbar(img1, ax=axs, orientation='vertical')

plt.show()