# -*- coding: utf-8 -*-
"""Split data into training and test"""

import numpy as np
from costs import *

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # Set seed
    np.random.seed(seed)
    
    # Shuffle data
    x = x[:, np.newaxis] if len(x.shape) < 2 else x
    data = np.append(x, y[:, np.newaxis], axis=1)
    np.random.shuffle(data)
    x, y = data[:, :-1], data[:, -1:]
    
    # Make sure there is at least one point in each set
    train_idx = int(ratio * len(y))
    if train_idx == 0:
        train_idx += 1
    elif train_idx == len(y) - 1:
        train_idx -= 1
    
    # Split
    train_x, train_y = x[:train_idx], y[:train_idx].ravel()
    test_x, test_y = x[train_idx:], y[train_idx:].ravel()
    
    return (train_x, train_y), (test_x, test_y)