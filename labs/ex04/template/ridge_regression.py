# -*- coding: utf-8 -*-

import numpy as np
from costs import compute_loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    lambda_prime = 2 * lambda_ * len(y)
    tx_t = tx.T

    w = np.dot(np.linalg.inv(tx_t.dot(tx) + lambda_prime * np.eye(tx.shape[1])), tx_t.dot(y))
    loss = compute_loss(y, tx, w, metric='mse')

    return w, loss