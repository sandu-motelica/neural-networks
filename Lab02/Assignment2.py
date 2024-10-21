import numpy as np
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score

def download_mnist(is_train: bool):
    dataset = MNIST(
        root='./Lab02/data',
        transform=lambda x: np.array(x).flatten(), 
        download=True,
        train=is_train
    )
    
    mnist_data = []
    mnist_labels = []
    
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    
    return np.array(mnist_data), np.array(mnist_labels)

def normalize_data(data):
    return data / 255.0

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def forward_propagation(X, W, b):
    z = np.dot(X, W) + b  
    y_pred = softmax(z) 
    return y_pred

def train_perceptron(X_train, y_train, W, b, epochs=50, batch_size=100, learning_rate=0.01):
    m = X_train.shape[0]

    for epoch in range(epochs):

        for i in range(0, m, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            y_pred = forward_propagation(X_batch, W, b)

            error = y_batch - y_pred

            grad_w = np.dot(X_batch.T, error) 
            grad_b = np.sum(error, axis=0)

            W += learning_rate * grad_w
            b += learning_rate * grad_b


    return W, b

def test_perceptron(X_test, y_test, W, b):
    y_pred = forward_propagation(X_test, W, b)
    predictions = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy



train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)

train_X = normalize_data(train_X)
test_X = normalize_data(test_X)

train_Y = one_hot_encode(train_Y)
test_Y = one_hot_encode(test_Y)

input_size = 784
output_size = 10
W = np.random.randn(input_size, output_size) * 0.01
b = np.zeros(output_size)

W_trained, b_trained = train_perceptron(train_X, train_Y, W, b, epochs=50, batch_size=100, learning_rate=0.01)

accuracy = test_perceptron(test_X, test_Y, W_trained, b_trained)
print(f"Test accuracy: {accuracy * 100:.2f}%")