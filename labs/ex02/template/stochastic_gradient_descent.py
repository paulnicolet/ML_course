# -*- coding: utf-8 -*-
from gradient_descent import compute_gradient
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w, error_type='mae'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w, error_type)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    ws = [initial_w]
    losses = []
    
    for step in range(max_iters):
        # Get one random mini batch
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            loss = compute_loss(batch_y, batch_tx, w)
            gradient = compute_stoch_gradient(batch_y, batch_tx, w)
            
            # Update gradient
            w = w - (gamma * gradient)
            
            # Store values
            ws.append(w)
            losses.append(loss)
        
    return losses, ws