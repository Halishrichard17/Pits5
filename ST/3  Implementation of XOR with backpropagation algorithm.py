import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR input and target data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_data = np.array([[0], [1], [1], [0]])

# Neural network architecture
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights randomly with mean 0
hidden_weights = np.random.uniform(size=(input_size, hidden_size))
output_weights = np.random.uniform(size=(hidden_size, output_size))

# Training loop
for _ in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(input_data, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_activation)

    # Calculate error
    error = target_data - predicted_output

    # Backpropagation
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_delta.dot(output_weights.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Update weights
    output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    hidden_weights += input_data.T.dot(hidden_layer_delta) * learning_rate

# Test the trained network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    hidden_layer_activation = np.dot(data, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_activation)
    print(f"Input: {data} Predicted Output: {predicted_output[0]}")




