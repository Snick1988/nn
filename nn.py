#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

import os
import cv2
import bz2
import time
import pickle
import numpy as np

from theano import tensor as T
from theano import function
from sklearn import decomposition
from sklearn.metrics import accuracy_score


class NeuralNet:

    settings = {
        'layers': [
            # (neurons per layer, activation function)
            (25, 'sigmoid'),
        ],
        'labels': 2,
        'epsilon': 1e-8,
        'alpha': 0.1,
        'lambda': 0.001,
        'momentum': 0.9,
        'resize': (128, 128),
        # Optimizations
        'with_gpu': False,
        'lr_optimizer': 'adam'
    }

    weights = []
    learning_rates = []
    t = 1

    def __init__(self, settings={}):
        self.settings.update(settings)

        ils = self.settings['resize'][0] * self.settings['resize'][1] * 3
        self.settings['layers'].insert(0, (ils, 'sigmoid'))

        for index in range(len(self.settings['layers'])):
            size, _ = self.settings['layers'][index]
            try:
                next_layer_size = self.settings['layers'][index + 1][0]
            except:
                next_layer_size = self.settings['labels']

            # Initialize weights for each layer
            self.weights.append(self.randweights(size, next_layer_size))

            # Initialize learning rate optimizations for each layer
            init = 0
            if self.settings['lr_optimizer'] == 'adam':
                init = (0, 0)
            self.learning_rates.append(init)

        # Initialize lr optimizer
        try:
            self.optimizer = getattr(self, self.settings['lr_optimizer'])
        except AttributeError:
            print('Invalid optimizer specified, using default (Nesterov) instead')
            self.optimizer = self.nesterov_momentum

        if self.settings['with_gpu']:
            x = T.dmatrix('x')
            y = T.dmatrix('y')
            f_dot = T.dot(x, y)
            f_elem = x * y

            tensor_dot = function([x, y], f_dot)
            tensor_elemwise = function([x, y], f_elem)

            self.dot = tensor_dot
            self.multiply = tensor_elemwise
        else:
            self.dot = np.dot
            self.multiply = lambda x, y: x * y

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x + self.settings['epsilon']))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def reLU(self, x):
        return np.maximum(x, 0, x)

    def reLU_prime(self, x):
        return 1. * (x > 0)

    def predict(self, x):
        """Predict output using learned weights"""
        self.bias = np.ones([x.shape[0], 1])
        for index, weight in enumerate(self.weights):
            if index == 0:
                h = self.sigmoid(self.dot(np.hstack([self.bias, x]), weight.T))
            else:
                h = self.sigmoid(self.dot(np.hstack([self.bias, h]), weight.T))

        loss = 1 - h.max(axis=1)

        return (loss, h.argmax(axis=1))

    def randweights(self, l_in, l_out):
        """Random weights initialization"""
        m = np.random.random((l_out, l_in + 1))
        return m * np.sqrt(2.0 / m)

    def cost(self, x, y):
        """Cost function"""

        self.bias = np.ones([1, x.shape[0]])

        activations, errors = self.forward(x)
        regularizations = []

        for weight in self.weights:
            regularizations.append(np.sum(weight[:, 1:] ** 2))

        Y = np.eye(self.settings['labels'])[y]

        # Number of examples
        m = self.bias.shape[1]

        # Forward prop cost
        eps = self.settings['epsilon']
        J = (1 / m) * np.sum(-Y * np.log(activations[-1] + eps).T - (1 - Y) * np.log(1 - activations[-1] + eps).T) + self.settings['lambda'] / (2 * m) * np.sum(regularizations)

        return J

    def grad(self, x, Y):
        """Compute gradient for hypothesis Theta"""

        activations, errors = self.forward(x)

        derivatives = []
        derivatives.append(activations[-1] - Y.T)

        # Compute derivative for each layer, except 1, starting from the last
        for index in range(1, len(self.settings['layers'])):
            drv_func = getattr(self, '{}_prime'.format(self.settings['layers'][index][1]))
            derivative = self.multiply(self.dot(self.weights[-index].T, derivatives[-index]), np.vstack([self.bias, drv_func(errors[-index])]))
            derivatives.insert(0, derivative[1:, :])

        derivatives[-1] = derivatives[-1].T
        # Remove bias from derivatives
        for index in range(len(derivatives) - 1):
            derivatives[index] = derivatives[index].T

        gradients = []
        # Number of examples
        m = self.bias.shape[1]

        for index, weight in enumerate(self.weights):
            weight_gradient = (1 / m) * self.dot(derivatives[index].T, activations[index].T)
            weight[0] = np.zeros([1, weight.shape[1]])
            gradient = weight_gradient + (self.settings['lambda'] / m) * weight

            gradients.append(gradient)

        return gradients

    def forward(self, x):
        """Forward propagation"""

        activations = []
        errors = []

        # Forward prop
        activations.append(np.vstack((self.bias, x.T)))

        # Compute errors for each activation
        for index, weight in enumerate(self.weights[:-1], start=1):
            act_func = getattr(self, self.settings['layers'][index][1])
            errors.append(self.dot(weight, activations[-1]))
            activations.append(np.vstack([self.bias, act_func(errors[-1])]))

        activations.append(self.sigmoid(self.dot(self.weights[-1], activations[-1])))

        return (activations, errors)

    def fit(self, x, y):
        """Training routine"""
        self.bias = np.ones([1, x.shape[0]])

        Y = np.eye(self.settings['labels'])[y]
        self.num_examples = x.shape[0]

        gradients = self.grad(x, Y)
        for index, gradient in enumerate(gradients):
            self.weights[index] += self.optimizer(gradient, index)

    def nesterov_momentum(self, gradient, index):
        momentum = self.settings['momentum'] * self.learning_rates[index] - self.settings['alpha'] * gradient
        coef = -self.settings['momentum'] * self.learning_rates[index] + (1 + self.settings['momentum']) * momentum

        # Save momentum for next iteration
        self.learning_rates[index] = momentum

        return coef

    def adam(self, gradient, index):
        m = 0.9 * self.learning_rates[index][0] + (1 - 0.9) * gradient
        v = 0.999 * self.learning_rates[index][1] + (1 - 0.999) * (gradient ** 2)

        # Bias correction
        mc = m / (1 - 0.9 ** self.t)
        vc = v / (1 - 0.999 ** self.t)
        self.t += 1

        # Save momentum for next iteration
        self.learning_rates[index] = (m, v)

        return -self.settings['alpha'] * mc / (np.sqrt(vc) + self.settings['epsilon'])

    def extract_features(self, file):
        """Extract features from image"""

        # Resize and subtract mean pixel
        img = cv2.resize(cv2.imread(file), self.settings['resize']).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

        # Normalize features
        img = (img.flatten() - np.mean(img)) / np.std(img)

        return np.array([img])

    def parse(self, path):
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

            X = self.extract_features(file)
            yield (X, np.array([classes.index(label)]))

    def save_checkpoint(self):
        with bz2.BZ2File('weights.pbz2', 'wb') as file:
            try:
                pickle.dump(self.weights, file, protocol=-1)
            # Make sure the file is actually written
            except KeyboardInterrupt:
                pass

    def load_checkpoint(self):
        with bz2.BZ2File('weights.pbz2', 'rb') as file:
            self.weigths = pickle.load(file)

    def load_validation(self, path):
        try:
            # Try loading validation set from file
            with bz2.BZ2File('valid.pbz2', 'rb') as file:
                X_valid, y_valid = pickle.load(file)

        except FileNotFoundError:
            # If loading fails - re-parse data and save to file
            X_valid = None
            y_valid = np.array([], int)

            for X, y in self.parse(path):
                if X_valid is None:
                    X_valid = np.array(X)
                else:
                    X_valid = np.vstack([X_valid, X])

                y_valid = np.append(y_valid, y)

            with bz2.BZ2File('valid.pbz2', 'wb') as file:
                pickle.dump([X_valid, y_valid], file, protocol=-1)

        return (X_valid, y_valid)

# Example usage
net = NeuralNet()
X_valid, y_valid = net.load_validation(os.path.join(os.getcwd(), 'dogscats', 'valid'))

try:
    net.load_checkpoint()
except:
    print('No saved weights found, will use random')

X, X_train = None, None
y, y_train = np.array([], int), np.array([], int)
cost = 10 ** 4

for epoch in range(200):

    if epoch > 0:
        current_cost = net.cost(X, y)
        loss, pred = net.predict(X_valid)
        score = accuracy_score(y_valid, pred)
        print('Pass: {0}; Accuracy: {1:.2f}%; Loss: {2:.2f}; Cost: {3:.6f}; Time spent: {4:.2f} seconds'.format(epoch, score * 100, np.sum(loss), current_cost, (time.time() - start)))

        # Increase learning rate by 10% if cost is decreasing
        if current_cost < cost:
            net.settings['alpha'] *= 1.1
        # Halve learning rate is cost is increasing
        else:
            net.settings['alpha'] /= 2.0

        cost = current_cost

    count = 0
    start = time.time()

    for X, y in net.parse(os.path.join(os.getcwd(), 'dogscats', 'train')):
        if X_train is None:
            X_train = np.array(X)
        else:
            X_train = np.vstack([X_train, X])

        y_train = np.append(y_train, y)

        if count % 50 == 0:
            net.fit(X_train, y_train)
            X_train = None
            y_train = np.array([], int)

        count += 1

    net.save_checkpoint()

loss, pred = net.predict(X_valid)
score = accuracy_score(y_valid, pred)
print('Final accuracy: {0:.2f}%; Loss: {1:.4f}'.format(score * 100, np.sum(loss)))
