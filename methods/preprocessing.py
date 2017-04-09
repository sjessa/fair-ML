import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from measures import discKC


def massage(X, y, S, b, d):
	"""Implementation of the 'massaging' data preprocessing technique
	from Kamiran and Calders (2012)

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)
		b: Protected value for sensitive attribute, 0 or 1
		d: Desired class, 0 or 1
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
	c = pr.columns.get_loc("class")
	pr.iloc[:M, c] = d
	dem.iloc[:M, c] = 1 - d

	X_prime = pd.concat([pr, dem, non]) 
	y_prime = X_prime['class'].tolist()
	X_prime = X_prime.drop(['class', 'prob'], axis = 1)

	return(X_prime, y_prime)


def main():

	data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
	X = data.drop(['crime'], axis = 1)
	y = np.asarray(data['crime'].tolist())

	X_prime, y_prime = massage(X, y, 'black', 1, 0)
	
	print(discKC(X, y, 'black', 1, 0)) # = 29.7%
	print(discKC(X_prime, y_prime, 'black', 1, 0)) # = almost 0

	X_prime['crime'] = y_prime
	X_prime.to_csv("output/crime_massaged.csv", index = False)

	test = pd.read_csv("test.csv")
	y_test = [1,1,1,1,0,0,0,1,0,1]

	print(discKC(test, y_test, 'sex', 0, 1)) # 40%
	a, b = massage(test, y_test, 'sex', 0, 1)
	print(discKC(a, b, 'sex', 0, 1)) # = 0


if __name__ == '__main__':
    main()