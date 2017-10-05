# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
	"""Compute loss using MSE or MAE."""
	e = y - tx.dot(w)

	#MAE
	#return (1.0 / len(y)) * np.sum(np.abs(e))

	# MSE
	return (1.0 / (2 * len(y))) * e.T.dot(e)