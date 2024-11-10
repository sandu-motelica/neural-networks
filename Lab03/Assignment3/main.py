from utils import *
from mlp import *
import numpy as np
from torchvision.datasets import MNIST

def preprocess_data():
    x_train, y_train = download_mnist(is_train=True)
    x_test, y_test = download_mnist(is_train=False)

    # normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)
    print(y_train.shape[0])
    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = preprocess_data()
    layer_sizes = [784, 100, 10]
    activation_functions = ["relu", "relu"]
    learning_rate = 0.01
    epochs = 20
    batch_size = 30
    patience = 5
    decay_factor = 0.5

    weights, biases = train_network(x_train, y_train, layer_sizes, activation_functions, learning_rate, epochs, batch_size, patience, decay_factor)

    # test set
    _, activations = forward_propagation(x_test, weights, biases, activation_functions)
    predictions = np.argmax(activations[-1], axis=1)
    targets = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == targets)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
