import numpy as np
import tensorflow as tf

# Define the non-linear function to approximate
def nonlinear_function(x1, x2, x3):
    return np.sin(x1) + np.cos(x2) * np.tanh(x3)

# Generate random input-output data for training
np.random.seed(0)
num_samples = 1000
X_train = np.random.uniform(-5, 5, size=(num_samples, 3))
y_train = nonlinear_function(X_train[:, 0], X_train[:, 1], X_train[:, 2])

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Test the trained model
X_test = np.random.uniform(-5, 5, size=(10, 3))
y_test = nonlinear_function(X_test[:, 0], X_test[:, 1], X_test[:, 2])
predictions = model.predict(X_test)

# Print the predictions and true values
for i in range(len(X_test)):
    print("Input:", X_test[i], "True Output:", y_test[i], "Predicted Output:", predictions[i][0])
