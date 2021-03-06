# -------------------------------------------------------------- #
#   Train three baseline classifiers on the preprocessed         #
#   Crime datasets                                               #
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

from baseline import learn_baselines


def main():

	# CRIME
	# 1. Massaged dataset
	data = pd.read_csv('output/crime_massaged.csv')
	data = shuffle(data)
	X = data.drop(['crime'], axis = 1)
	#X = X.as_matrix()
	y = np.asarray(data['crime'].tolist())

	learn_baselines(X, y, "massage")

	# 2. Massaged dataset
	data2 = pd.read_csv('output/crime_reweighed.csv')
	data2 = shuffle(data2)
	X2 = data2.drop(['crime'], axis = 1)
	#X2 = X2.as_matrix()
	y2 = np.asarray(data2['crime'].tolist())

	learn_baselines(X2, y2, "reweighed")

	# 3. Uniformly dataset
	data3 = pd.read_csv('output/crime_unisample.csv')
	data3 = shuffle(data3)
	X3 = data3.drop(['crime'], axis = 1)
	#X3 = X3.as_matrix()
	y3 = np.asarray(data3['crime'].tolist())

	learn_baselines(X3, y3, "unisample", run_svm = False)

	# RECIDIVISM
	# 1. Massaged dataset
	recid = pd.read_csv('output/recidivism_massaged.csv')
	recid = shuffle(recid)
	Xr = recid.drop(['recidivism'], axis = 1)
	#X = X.as_matrix()
	yr = np.asarray(recid['recidivism'].tolist())

	learn_baselines(Xr, yr, "recidivism-massage")

	# 2. Massaged dataset
	recid2 = pd.read_csv('output/recidivism_reweighed.csv')
	recid2 = shuffle(recid2)
	Xr2 = recid2.drop(['recidivism'], axis = 1)
	#X2 = X2.as_matrix()
	yr2 = np.asarray(recid2['recidivism'].tolist())

	learn_baselines(Xr2, yr2, "recidivism-reweighed")

	# 3. Uniformly dataset
	recid3 = pd.read_csv('output/recidivism_unisample.csv')
	recid3 = shuffle(recid3)
	Xr3 = recid3.drop(['recidivism'], axis = 1)
	#X3 = X3.as_matrix()
	yr3 = np.asarray(recid3['recidivism'].tolist())

	learn_baselines(Xr3, yr3, "recidivism-unisample", run_svm = False)



if __name__ == '__main__':
    main()