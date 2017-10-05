# -*- coding: utf-8 -*-
from costs import compute_loss

"""Gradient Descent"""

def compute_gradient_mse(y, tx, w):
    """Compute gradient for MSE."""
    e = y - tx.dot(w)
    return (-1.0 / len(y)) * tx.T.dot(e)

def compute_subgradient_mae(y, tx, w):
    """Compute one of the subgradient for MAE."""
    e = y - tx.dot(w)
    s = np.sign(e).reshape(-1, 1)
    return np.sum(-tx * s, axis=0) / len(y) 

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return compute_gradient_mse(y, tx, w)
    #return compute_subgradient_mae(y, tx, w)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # Update parameter vector
        w = w - (gamma * gradient)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws