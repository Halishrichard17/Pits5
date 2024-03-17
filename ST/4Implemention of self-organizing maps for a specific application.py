import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data (replace this with your own dataset)
np.random.seed(42)
data = np.random.rand(100, 2)

# SOM parameters
grid_size = (10, 10)  # Grid size of the SOM
input_dim = 2  # Dimensionality of the input data
learning_rate = 0.2
num_epochs = 1000

# Initialize the SOM
weight_matrix = np.random.rand(grid_size[0], grid_size[1], input_dim)

# Training loop
for epoch in range(num_epochs):
    for input_vector in data:
        # Find the Best Matching Unit (BMU)
        distances = np.linalg.norm(weight_matrix - input_vector, axis=-1)
        bmu_coords = np.unravel_index(np.argmin(distances), distances.shape)

        # Update the BMU and its neighbors
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_coords))
                influence = np.exp(-distance_to_bmu**2 / (2 * (epoch + 1)**2))  # Adjusting the influence based on the current epoch
                weight_matrix[i, j] += influence * learning_rate * (input_vector - weight_matrix[i, j])

# Create a map of cluster assignments
cluster_map = np.zeros((grid_size[0], grid_size[1]), dtype=int)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        distances = np.linalg.norm(data - weight_matrix[i, j], axis=-1)
        cluster_map[i, j] = np.argmin(distances)

# Visualize the results
plt.figure(figsize=(8, 8))
plt.pcolormesh(cluster_map, cmap='viridis')
plt.colorbar(label='Cluster')
plt.scatter(data[:, 0] * grid_size[0], data[:, 1] * grid_size[1], color='red', label='Data points')
plt.legend()
plt.title('Self-Organizing Map Clustering')
plt.show()


