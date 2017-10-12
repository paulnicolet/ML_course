import numpy as np
from costs import compute_loss

def least_squares(y, tx):
    """Calculate the least squares solution."""
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w, metric='mse')
    return w, loss