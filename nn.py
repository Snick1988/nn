#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

import os
import cv2
import bz2
import pickle
import numpy as np

from sklearn import decomposition
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))


def predict(theta1, theta2, x):
    """Predict output using learned weights"""
    m = x.shape[0]

    h1 = sigmoid(np.hstack((np.ones([m, 1]), x)).dot(theta1.T))
    h2 = sigmoid(np.hstack((np.ones([m, 1]), h1)).dot(theta2.T))

    loss = 1 - h2.max(axis=1)

    return (loss, h2.argmax(axis=1))


def randweights(l_in, l_out):
    """Random weights initialization"""
    m = np.random.random((l_out, l_in + 1))
    return m * np.sqrt(2.0 / m)


def unpack(params, ils, hls, labels):
    """Extract theta matrices from row vector"""
    theta1 = params[0: hls * (ils + 1)].reshape((hls, (ils + 1)))
    theta2 = params[(hls * (ils + 1)):].reshape((labels, (hls + 1)))

    return (theta1, theta2)


def cost(params, ils, hls, labels, x, y, lmbda=0.001):
    """Cost function"""

    theta1, theta2 = unpack(params, ils, hls, labels)
    a1, a2, a3, z2, m = forward(x, theta1, theta2)

    t1_reg = np.sum(theta1[:, 1:] ** 2)
    t2_reg = np.sum(theta2[:, 1:] ** 2)

    Y = np.eye(labels)[y]

    # Forward prop cost
    J = (1 / m) * np.sum(-Y * np.log(a3 + 1e-6).T - (1 - Y) * np.log(1 - a3 + 1e-6).T) + lmbda / (2 * m) * np.sum(t1_reg + t2_reg)

    return J


def grad(params, ils, hls, labels, x, y, lmbda=0.001):
    """Compute gradient for hypothesis Theta"""

    theta1, theta2 = unpack(params, ils, hls, labels)

    a1, a2, a3, z2, m = forward(x, theta1, theta2)

    Y = np.eye(labels)[y]
    d3 = a3 - Y.T

    d2 = np.dot(theta2.T, d3) * (np.vstack([np.ones([1, m]), sigmoid_prime(z2)]))
    d3 = d3.T
    d2 = d2[1:, :].T

    t1_grad = (1 / m) * np.dot(d2.T, a1.T)
    t2_grad = (1 / m) * np.dot(d3.T, a2.T)

    theta1[0] = np.zeros([1, theta1.shape[1]])
    theta2[0] = np.zeros([1, theta2.shape[1]])

    t1_grad = t1_grad + (lmbda / m) * theta1
    t2_grad = t2_grad + (lmbda / m) * theta2

    return np.concatenate([t1_grad.reshape(-1), t2_grad.reshape(-1)])


def forward(x, theta1, theta2):
    """Forward propagation"""

    m = x.shape[0]

    # Forward prop
    a1 = np.vstack((np.ones([1, m]), x.T))

    z2 = np.dot(theta1, a1)
    a2 = np.vstack((np.ones([1, m]), sigmoid(z2)))

    a3 = sigmoid(np.dot(theta2, a2))

    return (a1, a2, a3, z2, m)


def fit(x, y, t1, t2, labels=2, alpha=0.1, momentum=0):
    """Training routine"""
    ils = x.shape[1] if len(x.shape) > 1 else 1

    if t1 is None or t2 is None:
        t1 = randweights(ils, 200)
        t2 = randweights(200, labels)

    params = np.concatenate([t1.reshape(-1), t2.reshape(-1)])
    res = grad(params, ils, 200, labels, x, y)

    # c = cost(params, ils, 10, labels, x, y)

    # Attempt at Nesterov momentum
    momentum_prev = momentum
    momentum = 0.9 * momentum - alpha * res
    params += -0.9 * momentum_prev + (1 + 0.9) * momentum

    # params -= alpha * res
    t1, t2 = unpack(params, ils, 200, labels)

    return t1, t2, momentum


def extract(file):
    """Extract features from image"""

    # Resize and subtract mean pixel
    img = cv2.resize(cv2.imread(file), (128, 128)).astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    # Normalize features
    img = (img.flatten() - np.mean(img)) / np.std(img)

    return np.array([img])


def parse(path):
    """Parse data folder and return X and y"""
    examples = []
    for cwd, dirs, files in os.walk(path):
        if cwd == path:
            if not dirs:
                raise Exception('Invalid folder structure: expected folders divided by classes')

            classes = dirs
            continue

        current_class = os.path.basename(cwd)
        for file in [os.path.join(cwd, file) for file in files]:
            examples.append((file, current_class))

    np.random.shuffle(examples)

    for example in examples:
        file, label = example

        X = extract(file)
        yield (X, np.array([classes.index(label)]))

# Build and save validation dict
# X_valid = None
# y_valid = np.array([], int)
# for X, y in parse('/Users/snick/Downloads/dogscats/valid'):
#     if X_valid is None:
#         X_valid = np.array(X)
#     else:
#         X_valid = np.vstack([X_valid, X])

#     y_valid = np.append(y_valid, y)

# with bz2.BZ2File('valid.pbz2', 'wb') as file:
#     pickle.dump([X_valid, y_valid], file, protocol=-1)

# Load validation dict
with bz2.BZ2File('valid.pbz2', 'rb') as file:
    X_valid, y_valid = pickle.load(file)

t1, t2 = None, None
X_train = None
y_train = np.array([], int)
labels = len(set(y_valid))
momentum = 0

for epoch in range(30):
    if epoch > 0:
        params = np.concatenate([t1.reshape(-1), t2.reshape(-1)])
        c = cost(params, X.shape[1], 200, labels, X, y)
        loss, pred = predict(t1, t2, X_valid)
        score = accuracy_score(y_valid, pred)
        print('Pass: {0}; Accuracy: {1:.2f}%; Loss: {2:.2f}; Cost: {3:.6f}'.format(epoch, score * 100, np.sum(loss), c))

    count = 0
    for X, y in parse('/Users/snick/Downloads/dogscats/train'):
        if X_train is None:
            X_train = np.array(X)
        else:
            X_train = np.vstack([X_train, X])

        y_train = np.append(y_train, y)

        if count % 15 == 0:
            t1, t2, momentum = fit(X, y, t1, t2, labels=labels, momentum=momentum)
            X_train = None
            y_train = np.array([], int)

        count += 1

# Load pre-trained weights
# with bz2.BZ2File('weights.pbz2', 'rb') as file:
#     t1, t2 = pickle.load(file)

# Save weights
# with bz2.BZ2File('weights.pbz2', 'wb') as file:
#     pickle.dump([t1, t2], file, protocol=-1)

loss, pred = predict(t1, t2, X_valid)
score = accuracy_score(y_valid, pred)
print('Final accuracy: {0:.2f}%; Loss: {1:.4f}'.format(score * 100, np.sum(loss)))
