# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w, error_type='mae'):
	e = y - tx.dot(w)

	if error_type == 'mse':
		return (1.0 / (2 * len(y))) * e.T.dot(e)
	elif error_type == 'mae':
		return (1.0 / len(y)) * np.sum(np.abs(e))
	else:
		raise ValueError("error_type must be 'mse' or 'mae'")