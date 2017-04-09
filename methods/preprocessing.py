import numpy as np
import pandas as pd
import itertools
from sklearn.naive_bayes import GaussianNB

from measures import discKC


def massage(X, y, S, b, d):
	"""Implementation of the 'massaging' data preprocessing technique
	given by Algorithms 1 & 2 from Kamiran and Calders (2012)

	Flip the class labels of M pairs of examples in order where M
	is chosen explicitly to make the discKC(X, y) = 0. We choose the examples
	for which we flip labels using a ranker, here the ranker is a 
	Gaussian Naive Bayes classifier.

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)
		b: Protected value for sensitive attribute, 0 or 1
		d: Desired class, 0 or 1

	Returns:
		DataFrame: Training data (identical instances to X, but reordered)
		list: New binary class labels
	"""

	# Learn R, a Gaussian NB classifier which will act as a ranker
	R = GaussianNB()
	probas = R.fit(np.asarray(X), y).predict_proba(X)

	# Create a df with training data, labels, and desired class probabilities
	X['class'] = y
	X['prob'] = [record[d] for record in probas]

	# Promotion candidates sorted by descending probability of having desired class
	pr = X[(X[S] == b) & (X['class'] != d)]
	pr = pr.sort_values(by = 'prob', ascending = False)

	# Demotion candidates sorted by ascending probability
	dem = X[(X[S] != b) & (X['class'] == d)]
	dem = dem.sort_values(by = 'prob', ascending = True)

	# Non-candidates
	non = X[((X[S] == b) & (X['class'] == d)) | ((X[S] != b) & (X['class'] != d))]

    # Calculate the discrimination in the dataset
	disc = discKC(X, y, S, b, d)

	# Calculate M, the number of labels which need to be modified
	M = (disc * len(X[X[S] == b]) * len(X[X[S] != b])) / float(len(X))
	M = int(M)

	# Flip the class label of the top M objects of each group
	# i.e. M pairs swap labels, where M is chosen to make discKC = 0
	c = pr.columns.get_loc("class")
	pr.iloc[:M, c] = d
	dem.iloc[:M, c] = 1 - d

	X.drop(['class', 'prob'], axis = 1, inplace = True)
	X_prime = pd.concat([pr, dem, non]) 
	y_prime = X_prime['class'].tolist()
	X_prime = X_prime.drop(['class', 'prob'], axis = 1)

	return(X_prime, y_prime)


def reweigh(X, y, S):
	"""Implementation of the 'massaging' data preprocessing technique
	given by Algorithm 3 from Kamiran and Calders (2012)

	Add a new feature of weights to the training data in order to 
	make discKC(X, y) = 0

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)

	Return:
		DataFrame: New training data (identical instances to X, but with a new feature 'weight')
		list: Binary class labels (identical to y)

	"""

	X['label'] = y

	W = pd.DataFrame({'group': [1, 1, 0, 0], 'label': [1, 0, 1, 0]})

	# Calculate weight for each combination of sensitive attribute and class,
	# given by the expected probability of an example being in a certain group
	# and class if sensitive attribute/class are independent, divided by the
	# observed probability
	weights = [[len(X[X[S] == s]) * len(X[X['label'] == c]) \
				/ float(len(X) * len(X[(X[S] == s) & (X['label'] == c)])) \
				for c in [1, 0]]  for s in [1, 0]]

	W['weight'] = [i for j in weights for i in j]
	
	X_prime = X.copy()
	X_prime['weight'] = 0

	# Add weights according to class/group
	for s in [1, 0]:
		for c in [1, 0]:
			w = W.loc[(W['group'] == s) & (W['label'] == c), 'weight']
			X_prime.loc[(X[S] == s) & (X['label'] == c), 'weight'] = w.iloc[0]

	X.drop('label', axis = 1, inplace = True)
	y_prime = X_prime['label'].tolist()
	X_prime = X_prime.drop('label', axis = 1)

	return(X_prime, y_prime)



def uniform_sample(X, y, S, b, d):
	"""Implementation of the 'uniform sampling' data preprocessing technique
	given by Algorithms 4 from Kamiran and Calders (2012)

	Generate a new training dataset by uniformly sampling from the 
	input dataset, with the number of examples drawn from each group/class combo
	chosen to make discKC(X_prime, y) = 0 

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)
		b: Protected value for sensitive attribute, 0 or 1
		d: Desired class, 0 or 1

	Returns:
		DataFrame: New training data
		list: New binary class labels
	"""

	X['label'] = y

	W = pd.DataFrame({'group': [1, 1, 0, 0], 'label': [1, 0, 1, 0]})

	# Calculate weight for each combination of sensitive attribute and class,
	# given by the number of examples in each group divided by the number
	# that should be in each group if the data were non-discriminatory
	# NOTE: Algorithm 4 in the paper actually usees a denominator that appears to be wrong...
	weights = [[len(X[X[S] == s]) * len(X[X['label'] == c]) / float(len(X)*0.25) 
				# / float(len(X) * len(X[(X[S] == s) & (X['label'] == c)])) \
				for c in [1, 0]]  for s in [1, 0]]

	sizes = [[len(X[(X[S] == s) & (X['label'] == c)]) for c in [1, 0]] for s in [1, 0]]

	W['weight'] = [i for j in weights for i in j]
	W['size'] = [i for j in sizes for i in j]
	W = W.assign(num = lambda x: x.size * x.weight)

	# Divide the data into the four groups based on class/group
	dp = X[(X[S] == b) & (X['label'] == d)]
	dn = X[(X[S] == b) & (X['label'] != d)]
	fp = X[(X[S] != b) & (X['label'] == d)]
	fn = X[(X[S] != b) & (X['label'] != d)]

	# Uniformly sample from each group
	dp = dp.sample(n = W.loc[(W['group'] == b) & (W['label'] == d), 'num'].iloc[0].astype(int), replace = True)
	dn = dn.sample(n = W.loc[(W['group'] == b) & (W['label'] != d), 'num'].iloc[0].astype(int), replace = True)
	fp = fp.sample(n = W.loc[(W['group'] != b) & (W['label'] == d), 'num'].iloc[0].astype(int), replace = True)
	fn = fn.sample(n = W.loc[(W['group'] != b) & (W['label'] != d), 'num'].iloc[0].astype(int), replace = True)

	X_prime = pd.concat([dp, dn, fp, fn])
	X.drop('label', axis = 1, inplace = True)
	y_prime = X_prime['label'].tolist()
	X_prime = X_prime.drop('label', axis = 1)

	return(X_prime, y_prime)


def main():

    # On a little test dataset:
	X_test = pd.read_csv("test.csv")
	y_test = [1,1,1,1,0,0,0,1,0,1]

	print(discKC(X_test, y_test, 'sex', 0, 1)) # 40%
	a, b = massage(X_test, y_test, 'sex', 0, 1)
	print(discKC(a, b, 'sex', 0, 1)) # = 0

	c, d = reweigh(X_test, y_test, 'sex')

	e, f = uniform_sample(X_test, y_test, 'sex', 0, 1)
	print(discKC(e, f, 'sex', 0, 1)) # = 0

	# Crime dataset:
	data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
	X = data.drop(['crime'], axis = 1)
	y = np.asarray(data['crime'].tolist())

	X_prime, y_prime = massage(X, y, 'black', 1, 0)
	
	print(discKC(X, y, 'black', 1, 0)) # = 29.7%
	print(discKC(X_prime, y_prime, 'black', 1, 0)) # = almost 0

	X_prime['crime'] = y_prime
	X_prime.to_csv("output/crime_massaged.csv", index = False)

	X_prime2, y_prime2 = reweigh(X, y, 'black')
	X_prime2['crime'] = y_prime2
	X_prime2.to_csv("output/crime_reweighed.csv", index = False)

	X_prime3, y_prime3 = uniform_sample(X, y, 'black', 1, 0)
	print(discKC(X_prime3, y_prime3, 'black', 1, 0)) # = almost 0
	X_prime3['crime'] = y_prime3
	X_prime3.to_csv("output/crime_unisample.csv", index = False)



if __name__ == '__main__':
    main()