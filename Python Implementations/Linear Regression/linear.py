# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 02:32:32 2017

@author: heman
"""
import numpy as np, copy
import matplotlib.pyplot as plt

class LinearRegression:
    'Model class for the Linear Regression Model'
    
    def __init__(self):
        self.theta = None
        self.error = None
        self.scaling_info = None
    
    def train(self, data, targets, iter = 100,step = 1, lamda = 0):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        m, n = data.shape
        if self.scaling_info is None:
            self.scaling_info = np.zeros(n)
        data = copy.deepcopy(data)
        data = self.__scale_features(data.reshape(data.size), m, n)
        data = np.pad(data, ((0, 0), (1, 0)), mode = 'constant', constant_values = 1)
        if self.theta is None:
            self.theta = np.random.random((data.shape[1],))
        if self.error is None:
            self.error = np.zeros((iter, 2))
        for i in range(iter):
            predictions = data.dot(self.theta)
            self.error[i] = (i+1, self.__gradientDescent(data, targets, predictions, step, lamda))
            if not(i==0) and abs(self.error[i-1][1] - self.error[i][1]) < 1e-7:
                break
        self.error = self.error[0:i+1]
        self.__plot(xLabel='# of Iterations', yLabel='JTheta', title='Convergence Plot')
    
    def test(self, data, targets):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        m, n = data.shape
        data = copy.deepcopy(data)
        data = self.__scale_features(data.reshape(data.size), m, n)
        data = np.pad(data, ((0,0), (1,0)), mode='constant', constant_values = 1)
        predictions = data.dot(self.theta)
        errors = (predictions - targets)**2 / targets.shape[0]
        return (predictions, np.sqrt(errors))
    
    def __scale_features(self, data, m, n):
        for i in range(n):
            tmp = data[list(range(i, data.size, n))]
            if self.scaling_info[i] == 0:
                self.scaling_info[i] = max(tmp) - min(tmp)
            data[list(range(i, data.size, n))] = tmp / self.scaling_info[i]
            
        return data.reshape((m, n))
    
    def __gradientDescent(self, data, targets, predictions, step, lamda):
        diff = predictions - targets
        gradients = data.transpose().dot(diff)
        self.theta = self.theta * (1 - (step * lamda / targets.shape[0])) - (step * gradients / targets.shape[0])
        error = np.sqrt(sum(diff**2)) + np.sqrt(sum(self.theta**2)) * lamda
        return error / (2 * targets.shape[0])
    
    def __plot(self, error = None, xLabel='X-Label', yLabel='Y-Label', title='Plot'):
        if error is None: error = self.error
        fig = plt.figure()
        content = error.reshape(error.size)
        ax = fig.add_subplot(111)
        ax.plot(content[0: :2], content[1: :2])
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_title(title)
        plt.show()