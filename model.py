import numpy as np
import pickle
import os
import urllib.request
import tarfile
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
def load_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split("/")[-1]

    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()

    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def load_batch(folder, batch_id):
        file = os.path.join(folder, f'data_batch_{batch_id}')
        batch = unpickle(file)
        return batch[b'data'], batch[b'labels']

    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_batch('cifar-10-batches-py', i)
        train_data.append(data)
        train_labels += labels

    train_data = np.concatenate(train_data)
    test_batch = unpickle('cifar-10-batches-py/test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Reduce dataset size for testing
    train_data, train_labels = train_data[:1000], train_labels[:1000]
    test_data, test_labels = test_data[:200], test_labels[:200]

    return (train_data, np.array(train_labels)), (test_data, np.array(test_labels))

# One-hot encoding for labels
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# Normalize pixel data
def normalize_pixels(x):
    return x.astype(np.float32) / 255.0

# Initialize weights and biases for a layer
def init_weights(shape):
    return np.random.randn(*shape) * 0.01

def init_biases(shape):
    return np.zeros(shape)

# Convolution operation
def conv2d(X, W, b, stride=1, padding=0):
    X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    out_dim = ((X.shape[0] - W.shape[0] + 2 * padding) // stride + 1,
               (X.shape[1] - W.shape[1] + 2 * padding) // stride + 1,
               W.shape[3])
    output = np.zeros(out_dim)

    for i in range(out_dim[0]):
        for j in range(out_dim[1]):
            for k in range(W.shape[3]):
                region = X_padded[i*stride:i*stride+W.shape[0], j*stride:j*stride+W.shape[1], :]
                output[i, j, k] = np.sum(region * W[:, :, :, k]) + b[k]

    return output

# Max pooling operation
def max_pooling(X, pool_size=2, stride=2):
    out_dim = (X.shape[0] // stride, X.shape[1] // stride, X.shape[2])
    output = np.zeros(out_dim)

    for i in range(0, X.shape[0], stride):
        for j in range(0, X.shape[1], stride):
            for k in range(X.shape[2]):
                output[i // stride, j // stride, k] = np.max(X[i:i+pool_size, j:j+pool_size, k])

    return output

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Dropout
def dropout(X, keep_prob=0.5):
    mask = np.random.rand(*X.shape) < keep_prob
    return X * mask / keep_prob

# Batch normalization
def batch_norm(X, gamma, beta, eps=1e-5):
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(variance + eps)
    return gamma * X_norm + beta

# Flatten layer
def flatten(x):
    return x.flatten()

# Dense (fully connected) layer
def dense(x, W, b):
    return np.dot(x, W) + b

# Softmax activation function
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# Forward pass for the model
def forward_pass(x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta):
    x = conv2d(x, W_conv1, b_conv1, padding=1)
    x = batch_norm(x, gamma[0], beta[0])
    x = relu(x)
    x = max_pooling(x)

    x = conv2d(x, W_conv2, b_conv2, padding=1)
    x = batch_norm(x, gamma[1], beta[1])
    x = relu(x)
    x = max_pooling(x)

    x = conv2d(x, W_conv3, b_conv3, padding=1)
    x = batch_norm(x, gamma[2], beta[2])
    x = relu(x)
    x = max_pooling(x)

    x = flatten(x)
    x = dense(x, W_fc1, b_fc1)
    x = relu(x)
    x = dropout(x, keep_prob=0.5)  # Apply dropout
    x = dense(x, W_fc2, b_fc2)
    y_pred = softmax(x)

    return y_pred

# Evaluate model accuracy
def evaluate_accuracy(testX, testY, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta):
    correct_predictions = 0
    for i in range(len(testX)):
        x = testX[i].reshape((32, 32, 3))
        y_true = testY[i]

        y_pred = forward_pass(x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta)
        y_pred_label = np.argmax(y_pred)
        y_true_label = np.argmax(y_true)

        if y_pred_label == y_true_label:
            correct_predictions += 1

    accuracy = (correct_predictions / len(testX)) * 100
    return accuracy

# Model training loop
def train_model(trainX, trainY, testX, testY, epochs=10, lr=0.001):
    # Initialize weights and biases
    W_conv1 = init_weights((3, 3, 3, 32))
    b_conv1 = init_biases((32,))
    W_conv2 = init_weights((3, 3, 32, 64))
    b_conv2 = init_biases((64,))
    W_conv3 = init_weights((3, 3, 64, 128))
    b_conv3 = init_biases((128,))
    W_fc1 = init_weights((4*4*128, 256))  # Adjusted to match flattened size after pooling
    b_fc1 = init_biases((256,))
    W_fc2 = init_weights((256, 10))
    b_fc2 = init_biases((10,))

    gamma = [np.ones(32), np.ones(64), np.ones(128)]  # For batch normalization
    beta = [np.zeros(32), np.zeros(64), np.zeros(128)]  # For batch normalization

    for epoch in range(epochs):
        for i in range(len(trainX)):
            # Forward pass
            x = trainX[i].reshape((32, 32, 3))
            y = trainY[i]

            # Forward pass through the model
            y_pred = forward_pass(x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta)

            # Compute loss
            loss = cross_entropy_loss(y_pred, y)

            # Print progress
            if i % 100 == 0:  # Print progress every 100 samples
                print(f"Epoch {epoch + 1}/{epochs} - Processing sample {i+1}/{len(trainX)} - Loss: {loss}")

        # Evaluate accuracy after each epoch
        accuracy = evaluate_accuracy(testX, testY, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta)
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy}%")

    print("Training complete.")

# Test a specific image
def test_image(testX, testY, index, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta):
    x = testX[index].reshape((32, 32, 3))
    y_true = testY[index]

    # Forward pass through the model
    y_pred = forward_pass(x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, gamma, beta)

    # Get predicted and true labels
    y_pred_label = np.argmax(y_pred)
    y_true_label = np.argmax(y_true)

    # Display image and prediction
    plt.imshow(x)
    plt.title(f"True Label: {y_true_label}, Predicted Label: {y_pred_label}")
    plt.show()

# Main entry point
def main():
    # Load and preprocess data
    (trainX, trainY), (testX, testY) = load_cifar10()
    trainX, testX = normalize_pixels(trainX), normalize_pixels(testX)
    trainY, testY = one_hot_encode(trainY, 10), one_hot_encode(testY, 10)

    # Train the model
    train_model(trainX, trainY, testX, testY)

    # Initialize weights and biases for testing
    W_conv1 = init_weights((3, 3, 3, 32))
    b_conv1 = init_biases((32,))
    W_conv2 = init_weights((3, 3, 32, 64))
    b_conv2 = init_biases((64,))
    W_conv3 = init_weights((3, 3, 64, 128))
    b_conv3 = init_biases((128,))
    W_fc1 = init_weights((4*4*128, 256))  # Adjusted to match flattened size
    b_fc1 = init_biases((256,))
    W_fc2 = init_weights((256, 10))
    b_fc2 = init_biases((10,))

    gamma = [np.ones(32), np.ones(64), np.ones(128)]  # For batch normalization
    beta = [np.zeros(32), np.zeros(64), np.zeros(128)]  # For batch normalization

    # Test a specific image (index 0 for example)
    test_image(testX, testY, index=0, W_conv1=W_conv1, b_conv1=b_conv1, W_conv2=W_conv2, b_conv2=b_conv2, W_conv3=W_conv3, b_conv3=b_conv3, W_fc1=W_fc1, b_fc1=b_fc1, W_fc2=W_fc2, b_fc2=b_fc2, gamma=gamma, beta=beta)

if __name__ == "__main__":
    main()
