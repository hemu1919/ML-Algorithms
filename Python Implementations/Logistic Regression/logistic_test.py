# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:29:50 2017

@author: heman
"""
from request_data_link import get
import numpy as np
from logistic import LogisticRegression

link = 'http://data.princeton.edu/wws509/datasets/copen.raw'
m, n, parsed_data = get(link, 6)
index = list(range(0, parsed_data.size,6))
parsed_data = np.delete(parsed_data, index)
index = list(range(4, parsed_data.size, 5))
targets = parsed_data[index]
data = np.delete(parsed_data, index).reshape(m, n-1)
del link, m, n, parsed_data, index

regr = LogisticRegression()
regr.train(data, targets, iter=1000000, step = 0.001, lamda = 0)
labels, predictions = regr.test(data, targets)