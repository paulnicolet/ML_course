# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if len(x.shape) > 1:
    	if x.shape[1] > 1:
    		raise ValueError('x must be a vector')
    	x = x.ravel()
    	
    phi = np.zeros((degree + 1, len(x)))
    for d in range(degree + 1):
    	phi[d] = x**d

    return phi.T