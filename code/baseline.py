# -------------------------------------------------------------- #
#   Train three baseline classifiers on the Crime dataset           #
# -------------------------------------------------------------- #

import pandas as pd
import numpy as np
from scipy import interp

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_curve
from sklearn.svm import SVC


def main():

	data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
	data = shuffle(data)
	X = data.drop(['crime'], axis = 1)
	X = X.as_matrix()
	y = np.asarray(data['crime'].tolist())

	print("Now training a baseline Gaussian NB classifier...")
	gnb = GaussianNB()
	gnb_auc = baseline(gnb, "output/roc_gnb.csv", "output/predictions_gnb.csv", X, y)

	print("Now training a baseline SVM...")
	svm = SVC(kernel = 'linear', probability = True)
	svm_auc = baseline(svm, "output/roc_svm.csv", "output/predictions_svm.csv", X, y)

	print("Now training a baseline logistic regression model...")
	logr = LogisticRegression(penalty = 'l2')
	logr_auc = baseline(logr, "output/roc_logr.csv", "output/predictions_logr.csv", X, y)

	classifier = ["gnb", "svm", "logr"]
	auc = [gnb_auc, svm_auc, logr_auc]
	scores = pd.DataFrame({'classifier': classifier, 'auc': auc})
	scores.to_csv("output/auc_baseline.csv", index = False)


def baseline(classifier, roc_out, preds_out, X, y):

	preds = pd.DataFrame()
	roc = pd.DataFrame()

	cv = KFold(n_splits = 10, shuffle = False)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	for train, test in cv.split(X):
		X_train, X_test = X[train], X[test]
		y_train, y_test = y[train], y[test]

		classifier.fit(X_train, y_train)
		preds_k = classifier.predict(X_test)

		proba = classifier.predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
		proba = pd.DataFrame(proba)

		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0

		preds_df = pd.DataFrame({
			'neg': proba.iloc[:, 0],
			'pos': proba.iloc[:, 1],
			'pred': preds_k,
			'class': y_test})
		preds = preds_df.append(preds)

	roc = pd.DataFrame({'tpr': mean_tpr, 'fpr': mean_fpr})
	
	roc.to_csv(roc_out, index = False)
	preds.to_csv(preds_out, index = False)

	mean_tpr /= 10
	mean_tpr[-1] = 1.0
	auc_score = auc(mean_fpr, mean_tpr)
	print("AUC: " + str(auc_score))

	return(auc_score)


if __name__ == '__main__':
    main()