# -*- coding: utf-8 -*-
from costs import compute_loss

"""Gradient Descent"""

def compute_gradient(y, tx, w, error_type='mae'):
    e = y - tx.dot(w)

    if error_type == 'mse':
        return (-1.0 / len(y)) * tx.T.dot(e)
    elif error_type == 'mae':
        # Note: this computes one of the subgradient if MAE not differentiable in w
        return (-1 / len(y)) * (tx.T.dot(np.sign(e)))
    else:
        raise ValueError("error_type must be 'mse' or 'mae'")


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