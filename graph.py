import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os

# NOTE - this currently plots all of grid1 and all of grid2

# this is the 2D array that we will fill with the Grid data
arrayCopy = []
arrayOriginal = []

# fills arrayCopy with the copy grid data
with open("dataCopy.txt", "r") as f:
    for line in f.readlines():
        arrayCopy.append([float(num) for num in line.rstrip("\n ").split(" ")])

#fills arrayOriginal with the original grid data
with open("dataOriginal.txt", "r") as f:
    for line in f.readlines():
        arrayOriginal.append([float(num) for num in line.rstrip("\n ").split(" ")])


# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
bounds = [-1, -0.3, 0.3, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Plot arrayCopy in the first subplot
img1 = axs[0].imshow(arrayCopy, interpolation='nearest', cmap=cmap, norm=norm)
axs[0].set_title('Array Copy')

# Plot arrayOriginal in the second subplot
img2 = axs[1].imshow(arrayOriginal, interpolation='nearest', cmap=cmap, norm=norm)
axs[1].set_title('Array Original')

# Make a color bar for the first image (or you can make one for each if preferred)
fig.colorbar(img1, ax=axs, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1, 0, 1], orientation='vertical')

plt.show()