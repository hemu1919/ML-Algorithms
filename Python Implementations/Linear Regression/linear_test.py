# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:15:22 2017

@author: heman
"""

from linear import LinearRegression
from request_data_link import get
import numpy as np

link = 'http://data.princeton.edu/wws509/datasets/salary.raw'
m, n, parsed_data = get(link, 3)
index = list(range(n, parsed_data.size, n+1))
targets = parsed_data[index] / 10000;
data = np.delete(parsed_data, index).reshape(m, n)
data = np.hstack([data, data**2])
del link, m, n, parsed_data, index

regr = LinearRegression()
regr.train(data[0:round(0.8*data.shape[0])], targets[0:round(0.8*targets.shape[0])], iter=1000000, step = 0.1, lamda=1)
predictions, errors = regr.test(data[round(0.8*data.shape[0]):], targets[round(0.8*targets.shape[0]):])