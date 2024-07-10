import numpy as np

import matplotlib.pyplot as plt

# Read the matrix from the CSV file
matrix = np.genfromtxt('output_new.csv', delimiter=',')

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
# Show the plot in 2D
plt.show()

# Plot the matrix in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, matrix, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Temperature Grid in 3D')

# Show the 3D plot
plt.show()

import numpy as np

import matplotlib.pyplot as plt

# Read the matrix from the CSV file
matrix = np.genfromtxt('temperature_grid.csv', delimiter=',')

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
# Show the plot in 2D
plt.show()

# Plot the matrix in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, matrix, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Temperature Grid in 3D')

# Show the 3D plot
plt.show()