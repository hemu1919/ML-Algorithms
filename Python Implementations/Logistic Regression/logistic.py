# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:47:55 2017

@author: heman
"""
import numpy as np, copy, math
from matplotlib import pyplot as plt

class LogisticRegression:
    'Model class for Logistic Regression Model'
    
    def __init__(self):
        self.theta = None
        self.classes = None
        self.error = None
        self.scaling_info = None

    def train(self, data, targets, iter = 100, step = 1, lamda = 0):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        data = copy.deepcopy(data)
        if self.scaling_info is None:
            self.scaling_info = np.zeros(data.shape[1])
        m, n = data.shape
        self.__scale_features(data.reshape(m*n), m, n)
        data = np.pad(data, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        if self.classes is None:
            self.classes = np.unique(targets)
        if self.theta is None:
            self.theta = np.random.random((self.classes.size, data.shape[1]))
        if self.error is None:
            self.error = np.zeros((iter, self.classes.size + 1))
        for i in range(iter):
            predictions = data.dot(self.theta.transpose())
            predictions = np.array([self.__sigmoid(k) for k in predictions.reshape(predictions.size)]).reshape((data.shape[0], self.theta.shape[0]))
            self.error[i][1:] = self.__gradientDescent(data, targets, predictions.transpose(), step, lamda)
            self.error[i][0] = i+1
        self.error = self.error[0:i+1]
        self.__plot(xLabel='# of Iterations', yLabel='JTheta', title = "Convergence Plot for Class ")
    
    def test(self, data, targets):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        data = copy.deepcopy(data)
        m, n = data.shape
        data = self.__scale_features(data.reshape(data.size), m, n)
        data = np.pad(data, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        predictions = data.dot(self.theta.transpose())
        predictions = np.array([self.__sigmoid(k) for k in predictions.reshape(predictions.size)]).reshape((data.shape[0], self.theta.shape[0]))
        labels = [self.__classify(k, True) for k in predictions]
        return (labels, predictions)
    
    def __scale_features(self, data, m, n):
        for i in range(n):
            tmp = data[list(range(i, data.size, n))]
            if self.scaling_info[i] == 0:
                self.scaling_info[i] = max(tmp) - min(tmp)
            data[list(range(i, data.size, n))] = tmp / self.scaling_info[i]
        return data.reshape((m, n))
    
    def __plot(self, error = None, xLabel='X-Label', yLabel='Y-Label', title='Plot'):
        if error is None:
            error = self.error
        content = error.reshape(error.size);
        for i in range(self.error.shape[1] -1):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(content[0: :error.shape[1]], content[i+1: :error.shape[1]])
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_title(title + str(i + 1))
            plt.show()
    
    def __sigmoid(self, val):
        return 1 / (1 + math.exp(-val))
    
    def __classify(self, classes, return_class = False):
        index = np.argmax(classes)
        if return_class:
            index = self.classes[index]
        return index
    
    def __gradientDescent(self, data, targets, predictions, step, lamda):
        target_y = np.zeros((targets.shape[0], 1))
        error = np.zeros((1, self.classes.size))
        for i in range(self.classes.size):
            index = np.where(targets == self.classes[i])[0]
            target_y[index] = 1
            diff = predictions[i] - target_y.transpose();
            gradients = diff.dot(data)
            self.theta[i] = self.theta[i] * (1 - step * (lamda / target_y.size)) - (step * gradients / target_y.size)
            term1 = np.array([math.log(k) for k in predictions[i]])
            term2 = np.array([math.log(1-k) for k in predictions[i]])
            error[0][i] = (target_y.transpose().dot(term1) + (1 - target_y).transpose().dot(term2)[0] + (lamda * sum(self.theta[i]**2))) / target_y.size
            target_y[index] = 0
        return error
    