import numpy as np
import pandas as pd

def discKC(X, y, S, b, d):
	"""Implementation of the measure of discrimination in a labeled dataset
	or in predictions from a classifier from Definitions 1 and 2 in
	Kamiran and Calders (2012)

	This gives the difference in the probability of being in the desired class
	between the group with the protected value of the sensitive attribute
	(the marginalized group) and the favoured group. The desired class can
	be given by labeled training data, or by the prediction from a classifier.

	Args:
		X (DataFrame): Data
		y (list): Binary class labels or predictions for data
		S (str): Name of sensitive attribute (binary)
		b: Protected value for sensitive attribute, 0 or 1
		d: Desired class, 0 or 1
	"""

	D = X.copy()
	D['class'] = y

	# Fraction of eDamples with non-protected value for S in desired class d
	disc = len(D[(D[S] != b) & (D['class'] == d)]) / float(len(D[D[S] != b])) \
			- len(D[(D[S] == b) & (D['class'] == d)]) / float(len(D[D[S] == b]))

	return(disc)