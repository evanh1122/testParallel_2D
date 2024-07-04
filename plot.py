import numpy as np

import matplotlib.pyplot as plt

# Read the matrix from the CSV file
matrix = np.genfromtxt('output.csv', delimiter=',')

# Create a grid of indices
x = np.arange(matrix.shape[1])
y = np.arange(matrix.shape[0])
X, Y = np.meshgrid(x, y)

# Plot the matrix using a color gradient
plt.pcolormesh(X, Y, matrix, cmap='viridis')
plt.colorbar()

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Temperature Grid')

# Show the plot
plt.show()