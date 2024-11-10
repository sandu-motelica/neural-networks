import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train):
    dataset = MNIST(root='./Lab03/Assignment3/data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    data, labels = [], []
    for image, label in dataset:
        data.append(image)
        labels.append(label)

    return np.array(data), np.array(labels)

def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def he_initialization(size_in, size_out):
    std = np.sqrt(2 / size_in)
    return np.random.randn(size_in, size_out) * std

def xavier_initialization(size_in, size_out):
    limit = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-limit, limit, (size_in, size_out))
