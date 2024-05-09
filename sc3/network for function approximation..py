import numpy as np
import matplotlib.pyplot as plt

# Define the function to approximate
def function_to_approximate(x):
    return np.sin(x) + 0.5 * x

# Generate data for training
np.random.seed(0)
num_samples = 1000
X_train = np.random.uniform(-5, 5, size=(num_samples, 1))
y_train = function_to_approximate(X_train)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.randn(1, 32)
        self.bias1 = np.zeros((1, 32))
        self.weights2 = np.random.randn(32, 32)
        self.bias2 = np.zeros((1, 32))
        self.weights3 = np.random.randn(32, 1)
        self.bias3 = np.zeros((1, 1))
    
    def forward(self, X):
        self.layer1 = np.maximum(0, np.dot(X, self.weights1) + self.bias1)
        self.layer2 = np.maximum(0, np.dot(self.layer1, self.weights2) + self.bias2)
        self.output = np.dot(self.layer2, self.weights3) + self.bias3
        return self.output

# Train the neural network
learning_rate = 0.001
epochs = 50
batch_size = 32

network = NeuralNetwork()
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        output = network.forward(X_batch)
        
        # Compute loss (mean squared error)
        loss = np.mean((output - y_batch) ** 2)
        
        # Backpropagation
        output_grad = 2 * (output - y_batch) / len(X_batch)
        weights3_grad = np.dot(network.layer2.T, output_grad)
        bias3_grad = np.sum(output_grad, axis=0, keepdims=True)
        layer2_grad = np.dot(output_grad, network.weights3.T)
        layer2_grad[network.layer2 <= 0] = 0
        weights2_grad = np.dot(network.layer1.T, layer2_grad)
        bias2_grad = np.sum(layer2_grad, axis=0, keepdims=True)
        layer1_grad = np.dot(layer2_grad, network.weights2.T)
        layer1_grad[network.layer1 <= 0] = 0
        weights1_grad = np.dot(X_batch.T, layer1_grad)
        bias1_grad = np.sum(layer1_grad, axis=0, keepdims=True)
        
        # Update weights and biases
        network.weights1 -= learning_rate * weights1_grad
        network.bias1 -= learning_rate * bias1_grad
        network.weights2 -= learning_rate * weights2_grad
        network.bias2 -= learning_rate * bias2_grad
        network.weights3 -= learning_rate * weights3_grad
        network.bias3 -= learning_rate * bias3_grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the trained network
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
predictions = network.forward(X_test)

# Plot the results
plt.plot(X_test, function_to_approximate(X_test), label='True Function')
plt.plot(X_test, predictions, label='Predictions')
plt.title('Function Approximation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
