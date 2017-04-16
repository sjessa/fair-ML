# -------------------------------------------------------------- #
#                                                                #
#   Implement the 2NB model proposed by Calders and Verwer (2010)  #
#   which splits the dataset on the sensitive attribute and      #
#   trains a separate NB model for each.                         #
#                                                                #
# -------------------------------------------------------------- #

import pandas as pd
import numpy as np
from scipy import interp

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_curve

from measures import discKC


def main():

	data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
	data = shuffle(data)
	X = data.drop(['crime'], axis = 1)
	y = np.asarray(data['crime'].tolist())

	print(discKC(X, y, 'black', 1, 0))
	y_hat = two_nb(X, y, 'black')
	print(discKC(X, y_hat.tolist(), 'black', 1, 0))


def split_on_sensitive_attribute(X, y, S):
	"""Given a dataset and its labels, split the data on the value of
	a binary sensitive attribute. Here, let X+ denote the subset
	of the data with a positive value for S, and X- note its complement.

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)

	Returns:
		DataFrame: X+
		list: Labels for X+
		DataFrame: X-
		list: Labels for X-
	"""

	idx_pos = np.where(X[S] == 1)[0]
	X_pos, y_pos = X.iloc[idx_pos], y[idx_pos]
	X_neg, y_neg = X.drop(X.index[idx_pos]), np.delete(y, idx_pos)

	return(X_pos, y_pos, X_neg, y_neg)



def two_nb(X, y, S):
	"""Implementation of the two Naive Bayes models method for reducing discrimination
	in classification proposed by Calders and Verwer (2010)

	The dataset is split on the value of the sensitive attribute, and a 
	separate NB model learned for each. To predict on a new example, apply
	the model corresponding to its value for the sensitive attribute.

	Args:
		X (DataFrame): Training data
		y (list): Binary class labels for training data
		S (str): Name of sensitive attribute (binary)

	Returns:
		list: Binary class predictions

	"""

	preds = pd.DataFrame()
	roc = pd.DataFrame()

	cv = KFold(n_splits = 10, shuffle = False)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	gnb_pos = GaussianNB()
	gnb_neg = GaussianNB()

	for train, test in cv.split(X):
		X_train, X_test = X.iloc[train], X.iloc[test]
		y_train, y_test = y[train], y[test]

		# For every train/test split, we further partition each dataset
		# on the value of the sensitive attribute
		X_train_pos, y_train_pos, X_train_neg, y_train_neg = split_on_sensitive_attribute(X_train, y_train, S)
		X_test_pos, y_test_pos, X_test_neg, y_test_neg = split_on_sensitive_attribute(X_test, y_test, S)

		# Train a separate NB classifier on each training set
		gnb_pos.fit(X_train_pos, y_train_pos)
		gnb_neg.fit(X_train_neg, y_train_neg)

		# And predict on the test set using the corresponding model,
		# then combine the predictions
		preds_k_pos = gnb_pos.predict(X_test_pos).tolist()
		preds_k_neg = gnb_neg.predict(X_test_neg).tolist()
		preds_k = preds_k_pos + preds_k_neg

		proba_pos = pd.DataFrame(gnb_pos.predict_proba(X_test_pos))
		proba_neg = pd.DataFrame(gnb_neg.predict_proba(X_test_neg))
		proba = pd.concat([proba_pos, proba_neg])

		y_test_reordered = y_test_pos.tolist() + y_test_neg.tolist()
		fpr, tpr, thresholds = roc_curve(y_test_reordered, proba.iloc[:,1].values)

		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0

		preds_df = pd.DataFrame({
			'neg': proba.iloc[:, 0],
			'pos': proba.iloc[:, 1],
			'pred': preds_k,
			'class': y_test_reordered})
		preds = preds_df.append(preds)

	mean_tpr /= 10
	roc = pd.DataFrame({'tpr': mean_tpr, 'fpr': mean_fpr})
	
	roc.to_csv("output/2nb_roc_gnb.csv", index = False)

	X_out = X.copy()
	new_idx = range(0, len(X.index))
	X_out['idx'] = new_idx
	X_out = X_out.set_index(['idx'], drop = True)

	X_out = X_out.join(preds)
	X_out.to_csv("output/2nb_predictions_gnb.csv", index = False)

	#preds.to_csv(, index = False)

	mean_tpr[-1] = 1.0
	auc_score = auc(mean_fpr, mean_tpr)
	print("AUC: " + str(auc_score))

	classifier = ["2nb"]
	auc_score = [auc_score]
	scores = pd.DataFrame({'classifier': classifier, 'auc': auc_score})
	scores.to_csv("output/2nb_auc.csv", index = False)

	return(preds['pred'])


if __name__ == '__main__':
    main()