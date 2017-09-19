# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:27:02 2017

@author: heman
"""
import requests, numpy as np

def get(link = None, times = 1):
    text = '';
    if link is None: return text
    f = requests.get(link)
    text = f.text
    f.close()
    return parse(text, times)
def parse(raw_data, times):
    spaces = " " * times
    raw_data = raw_data.split("\r\n")
    raw_data = [str1.split(spaces) for str1 in [str2.strip() for str2 in raw_data]][0:-1]
    m, n = len(raw_data), len(raw_data[0]) - 1
    parsed_data = np.array(raw_data, dtype = 'float64').reshape(m * (n + 1))
    return m, n ,parsed_data