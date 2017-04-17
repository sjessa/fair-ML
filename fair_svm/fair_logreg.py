import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from scipy import interp
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_data():
    data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
    data['id'] = data.index

    data = shuffle(data)
    shuffled_id_arr = data['id']

    X = data.drop(['crime', 'black', 'id'], axis = 1)
    X = X.as_matrix()

    n, dim = X.shape
    # append bias term to X
    X = np.concatenate((X, np.ones((n, 1))), axis=1)

    s = np.asarray(data['black'].tolist())
    y = np.asarray(data['crime'].tolist())
    return X, s, y, shuffled_id_arr


class log_reg_acc_const:
    """
    Logistic regression as a constrained optimization problem, where the loss function
    is the covariance between the sensitive attribute and the signed distance from the decision boundary.
    The constraint is on the log-likelihood loss function.
    """
    def __init__(self, gamma):
        self.w = None
        self.gamma = gamma
        self.baseline_w = pd.read_csv('baselinelogr_weights.csv').as_matrix()
        self.baseline_loss = 492

    def fit(self, X, s, y):
        n, dim = X.shape

        y = y.reshape(n, 1)
        s = s.reshape(n, 1) - np.mean(s)*np.ones((n, 1))

        G = (1 / n) * s.T.dot(X)

        def objective(w, sign=1.0):
            return np.absolute(G.dot(w))

        def grad_objective(w, sign=1.0):
            return np.sign(G.dot(w))*G

        def log_likelihood_const(w):
            p = sigmoid(X.dot(w)).reshape(n, 1)
            ones = np.ones((n, 1))
            f = -np.sum(y * np.log(p) + (ones - y) * np.log(ones - p), axis=0)
            return (1+self.gamma)*self.baseline_loss - f

        def grad_loglikelihood_const(w):
            p = sigmoid(X.dot(w)).reshape(n, 1)
            # using numpy broadcasting, each row of X is multiplied by (y-p),
            # the sum is taken over the rows to produce the transpose of the gradient
            Df = np.sum(X * (y - p), axis=0)
            return Df

        cons = ({'type': 'ineq',
             'fun': log_likelihood_const,
             'jac': grad_loglikelihood_const})

        res = minimize(objective, self.baseline_w, jac=grad_objective,
               constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})

        self.w = res.x

    def predict(self, X):
        return np.round(sigmoid(X.dot(self.w)))


class log_reg_fairness_const:

    """
    logistic regression as a constrained optimization problem where the objective is the log-loss function,
    and the constraint is the covariance of the sensitive attribute s with the distance from the boundary
    :param c: upper bound on the covariance
    :return:
    """

    def __init__(self, c):
        self.w = None
        self.c = c
        self.baseline_w = pd.read_csv('baselinelogr_weights.csv').as_matrix()

    def fit(self, X, s, y):
        n, dim = X.shape

        y = y.reshape(n, 1)
        s = s.reshape(n, 1) - np.mean(s)*np.ones((n, 1))

        def log_likelihood(w, sign=1.0):
            p = sigmoid(X.dot(w)).reshape(n, 1)
            ones = np.ones((n, 1))
            f = -np.sum(y*np.log(p)+(ones-y)*np.log(ones-p), axis=0)
            return f

        def grad_loglikelihood(w, sign=1.0):
            p = sigmoid(X.dot(w)).reshape(n, 1)
            # using numpy broadcasting, each row of X is multiplied by (y-p),
            # the sum is taken over the rows to produce the transpose of the gradient
            Df = -np.sum(X*(y-p), axis=0)
            return Df

        # matrix for linear constraint used for training
        G = (1 / n) * s.T.dot(X)

        cons = ({'type': 'ineq',
                 'fun': lambda w: G.dot(w) + c,
                 'jac': lambda w: G},
                {'type': 'ineq',
                 'fun': lambda w: c - G.dot(w),
                 'jac': lambda w: -G})

        res = minimize(log_likelihood, self.baseline_w, jac=grad_loglikelihood,
               constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})

        self.w = res.x

    def predict(self, X):
        return np.round(sigmoid(X.dot(self.w)))

def cross_val(classifier, run):
    X, s, y, shuffled_id_arr = load_data()

    cv = KFold(n_splits=10, shuffle=False)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    pred = pd.DataFrame(columns=['id', 'black', 'actual', 'prediction'])
    pred['id'] = shuffled_id_arr
    pred['actual'] = y
    pred['black'] = s
    split = 0
    for train, test in cv.split(X):
        split+=1
        X_train, X_test = X[train, :], X[test, :]
        s_train, s_test = s[train], s[test]
        y_train, y_test = y[train], y[test]
        test_id_arr = shuffled_id_arr[test]

        classifier.fit(X_train, s_train, y_train)
        preds_k = classifier.predict(X_test)
        pred.loc[pred['id'].isin(test_id_arr), 'prediction'] = preds_k
        fpr, tpr, thresholds = roc_curve(y_test, preds_k)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    pred.to_csv('fairness_const_pred/fairness_const_test_set_predictions_'+str(run)+'.csv')
    roc = pd.DataFrame({'tpr': mean_tpr, 'fpr': mean_fpr})

    plt.scatter(roc['tpr'], roc['fpr'])
    mean_tpr /= 10
    mean_tpr[-1] = 1.0
    auc_score = auc(mean_fpr, mean_tpr)
    print("AUC: " + str(auc_score))

    return (auc_score)


if __name__=="__main__":
    C = 0.1
    for run in range(10):
        c = np.array([C])
        classifier = log_reg_fairness_const(c)
        cross_val(classifier, run)
        C += 0.1

    plt.show()