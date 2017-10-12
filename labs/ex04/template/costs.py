# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_e(y, tx, w):
	return y - tx.dot(w)

def compute_loss(y, tx, w, metric='mae'):
	e = compute_e(y, tx, w)

	if metric == 'mse':
		return (1.0 / (2 * len(y))) * e.T.dot(e)
	elif metric == 'mae':
		return (1.0 / len(y)) * np.sum(np.abs(e))
	else:
		raise ValueError("metric must be 'mse' or 'mae'")