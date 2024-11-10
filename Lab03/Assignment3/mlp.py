import numpy as np
from utils import *


def initialize_parameters(layer_sizes, activation_functions):
    weights, biases = [], []

    for i in range(len(layer_sizes) - 1):
        size_in, size_out = layer_sizes[i], layer_sizes[i + 1]
        if activation_functions[i] == "sigmoid":
            weights.append(xavier_initialization(size_in, size_out))
        elif activation_functions[i] == "relu":
            weights.append(he_initialization(size_in, size_out))
        else:
            weights.append(np.random.randn(size_in, size_out) * 0.01)
        biases.append(np.zeros((1, size_out)))

    return weights, biases


def forward_propagation(X, weights, biases, activation_functions):
    activations = [X]
    z_values = []

    for i in range(len(weights) - 1):
        z = activations[-1] @ weights[i] + biases[i]
        z_values.append(z)
        if activation_functions[i] == "relu":
            activations.append(relu(z))
        elif activation_functions[i] == "sigmoid":
            activations.append(sigmoid(z))
        else:
            raise ValueError(f"Invalid activation function")

    # output layer with softmax
    z = activations[-1] @ weights[-1] + biases[-1]
    z_values.append(z)
    output = softmax(z)
    activations.append(output)

    return z_values, activations


def backpropagation(X, y, weights, biases, z_values, activations, activation_functions, learning_rate):
    m = y.shape[0]
    delta = activations[-1] - y
    d_weights, d_biases = [None] * len(weights), [None] * len(biases)

    # gradients for the output layer
    d_weights[-1] = activations[-2].T @ delta / m
    d_biases[-1] = np.sum(delta, axis=0, keepdims=True) / m

    # backpropagate through hidden layers
    for i in reversed(range(len(weights) - 1)):
        if activation_functions[i] == "relu":
            delta = (delta @ weights[i + 1].T) * relu_derivative(z_values[i])
        else:
            delta = (delta @ weights[i + 1].T) * sigmoid_derivative(z_values[i])
        d_weights[i] = activations[i].T @ delta / m
        d_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

    # update weights and biases
    for i in range(len(weights)):
        weights[i] -= learning_rate * d_weights[i]
        biases[i] -= learning_rate * d_biases[i]

    return weights, biases


def train_network(X, y, layer_sizes, activation_functions, learning_rate, epochs, batch_size, patience, decay_factor):
    weights, biases = initialize_parameters(layer_sizes, activation_functions)
    best_accuracy = 0
    no_improvement_count = 0

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            z_values, activations = forward_propagation(X_batch, weights, biases, activation_functions)
            weights, biases = backpropagation(X_batch, y_batch, weights, biases, z_values, activations, activation_functions, learning_rate)

        # compute training accuracy
        _, activations = forward_propagation(X, weights, biases, activation_functions)
        predictions = np.argmax(activations[-1], axis=1)
        targets = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == targets)
        print(f'Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy * 100:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # decay the learning rate if no improvement is seen for `patience` epochs
        if no_improvement_count >= patience:
            learning_rate *= decay_factor
            print(f"Learning rate reduced to {learning_rate:.5f}")
            no_improvement_count = 0

    return weights, biases
