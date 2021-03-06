#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

import os
import cv2
import bz2
import time
import pickle
import numpy as np
import argparse

from theano import function, tensor as T
from sklearn.metrics import accuracy_score


class NeuralNet:

    settings = {
        'layers': [
            # (neurons per layer, activation function)
            (800, 'sigmoid'),
        ],
        'alpha': 0.000001,
        'batch': 10,
        'epochs': 400,
        'epsilon': 1e-8,
        'labels': 2,
        'lambda': 1,
        'momentum': 0.9,
        'resize': (124, 124),
        # Optimizations
        'with_gpu': False,
        'lr_optimizer': 'adam'
    }

    weights = []
    learning_rates = []
    t = 1

    def __init__(self, settings={}):

        self.settings.update(settings)
        self.load_validation()

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

        try:
            self.load_checkpoint()
        except:
            print('No saved weights found, will use random')

        # Initialize lr optimizer
        try:
            self.optimizer = getattr(self, self.settings['lr_optimizer'])
        except AttributeError:
            print('Invalid optimizer specified, using default (Nesterov) instead')
            self.optimizer = self.nesterov_momentum

        if self.settings['with_gpu']:
            x = T.dmatrix('x')
            y = T.dmatrix('y')

            self.dot = function([x, y], T.dot(x, y))
            self.multiply = function([x, y], x * y)
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

        # Compute derivative for each layer, except input, starting from the last
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
            # Yield features and label in training mode
            if self.settings['mode'] == 'train':
                if cwd == path:
                    if not dirs:
                        raise Exception('Invalid folder structure: expected folders divided by classes')

                    classes = dirs
                    continue

                current_class = os.path.basename(cwd)
                for file in [os.path.join(cwd, file) for file in files]:
                    examples.append((file, current_class))

                if not examples:
                    raise Exception('No files found in {}'.format(path))

                np.random.shuffle(examples)

                for example in examples:
                    file, label = example

                    X = self.extract_features(file)
                    yield (X, np.array([classes.index(label)]))

            else:
                # Yield features in testing mode
                if cwd == path:
                    continue

                for file in [os.path.join(cwd, file) for file in files]:
                    yield (file, self.extract_features(file))

    def save_checkpoint(self):
        with bz2.BZ2File('weights.pbz2', 'wb') as file:
            try:
                pickle.dump(self.weights, file, protocol=-1)
            # Make sure the file is actually written
            except KeyboardInterrupt:
                pass

    def load_checkpoint(self):
        with bz2.BZ2File('weights.pbz2', 'rb') as file:
            weights = pickle.load(file)

        if sum([w[0] for w in self.weights]) != sum([len(w) for w in weight]):
            print('Saved weights do not match current settings. Will use random.')
            return True

        self.weights = weights

    def load_validation(self):
        try:
            # Try loading validation set from file
            with bz2.BZ2File('valid.pbz2', 'rb') as file:
                X_valid, y_valid = pickle.load(file)

        except FileNotFoundError:
            # If loading fails - re-parse data and save to file
            X_valid = None
            y_valid = np.array([], int)

            for X, y in self.parse(self.settings['validation']):
                if X_valid is None:
                    X_valid = np.array(X)
                else:
                    X_valid = np.vstack([X_valid, X])

                y_valid = np.append(y_valid, y)

            with bz2.BZ2File('valid.pbz2', 'wb') as file:
                pickle.dump([X_valid, y_valid], file, protocol=-1)

        self.x_valid = X_valid
        self.y_valid = y_valid

        return True

    def run(self):
        """Main routine - train or predict"""

        if self.settings['mode'] == 'train':
            y_train = np.array([], int)
            x_train = None
            for epoch in range(self.settings['epochs']):
                # Init per-epoch defaults
                count = 0
                start = time.time()
                self.t = 1

                for x, y in self.parse(self.settings['train']):
                    x_train = np.vstack([x_train, x]) if x_train is not None else np.array(x)
                    y_train = np.append(y_train, y)

                    if count % self.settings['batch'] == 0:
                        self.fit(x_train, y_train)
                        x_train = None
                        y_train = np.array([], int)

                    count += 1

                else:
                    # Process last batch if it exists
                    if x_train is not None:
                        self.fit(x_train, y_train)
                        loss, pred = self.predict(x_train)
                        score = accuracy_score(y_train, pred)
                        print('Training batch accuracy: {:.2f}%; Loss: {:.2f}'.format(score * 100, np.sum(loss)))

                cost = self.cost(x, y)
                loss, pred = self.predict(self.x_valid)
                score = accuracy_score(self.y_valid, pred)
                log_data = [epoch + 1, score * 100, np.sum(loss), cost, (time.time() - start), self.settings['alpha']]

                print('Pass: {}; Accuracy: {:.2f}%; Loss: {:.2f}; Cost: {:.6f}; Time spent: {:.2f} seconds; Learning rate: {:.10f}'.format(*log_data))
                self.save_checkpoint()

        elif self.settings['mode'] == 'test':
            for file, x in self.parse(self.settings['test']):
                loss, pred = self.predict(x)
                print('File: {}, Prediction: {}; Loss: {:.2f}'.format(file, pred, loss))

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Path to training set")
parser.add_argument("-v", "--validation", help="Path to validation set")
parser.add_argument("-ts", "--test", help="Path to testing set")
parser.add_argument("-m", "--mode", help="Operation mode", choices=['train', 'test'], default='train')

args = parser.parse_args()
if args.mode == 'train' and (not args.train or not args.validation):
    raise SystemExit('You must provide paths to training and validation sets for "train" mode')

elif args.mode == 'test' and not args.test:
    raise SystemExit('You must provide path to testing set for "test" mode')

net = NeuralNet(vars(args))
net.run()
