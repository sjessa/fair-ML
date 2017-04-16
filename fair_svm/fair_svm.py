import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from scipy import interp

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def load_data():
    data = pd.read_csv('../data/crime/crime_clean.tsv', sep = '\t')
    data = shuffle(data)
    X = data.drop(['crime', 'black'], axis = 1)
    X = X.as_matrix()
    s = np.asarray(data['black'].tolist())
    y = np.asarray(data['crime'].tolist())
    return X, s, y


# class svm_fairness_const:
#
#     def __init__(self, svm_c, cov):
#         self.w = None
#         self.svm_c = svm_c
#         self.baseline_w = None
#         self.cov = cov
#
#     def fit(self, X, s, y):
#         n, dim = X.shape
#         self.baseline_w = np.zeros((n, 1))
#         y = y.reshape(n, 1)
#         s = s.reshape(n, 1) - np.mean(s) * np.ones((n, 1))
#         G = (1 / n) * s.T.dot(X)
#
#         def objective(zeta_w, sign=1.0):
#             p = self.svm_c*np.ones((1, n))
#             return p.dot(zeta_w[0:n]) + 0.5*zeta_w[n:].T.dot(zeta_w[n:])
#
#         def grad_objective(zeta_w, sign=1.0):
#             grad_w = zeta_w[n:].reshape(1, n)
#             grad_zeta = self.svm_c*np.ones(1, n)
#             return np.concatenate((grad_zeta, grad_w), axis=0)
#
#         def cov_const_pos(alpha):
#             self.w = (y*X).T.dot(alpha)
#             return G.dot(self.w) + self.cov
#
#         def cov_const_neg(alpha):
#             self.w = (y*X).T.dot(alpha)
#             return self.cov - G.dot(self.w)
#
#         cons = ({'type': 'ineq',
#                  'fun': lambda w: G.dot(w) + c,
#                  'jac': lambda w: G},
#                 {'type': 'ineq',
#                  'fun': lambda w: c - G.dot(w),
#                  'jac': lambda w: -G},
#                 {'type': 'ineq',
#                 'fun': np.ones((n, 1)) - y*X.dot(w)
#         )
#
#         res = minimize(dual_objective, self.baseline_alpha, jac=grad_objective,
#                constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})
#
#         self.w = res.x
#
#     def predict(self, X):
#         return sigmoid(X.dot(self.w))


class svm_fairness_const:

    def __init__(self, svm_c, cov):
        self.w = None
        self.svm_c = svm_c
        self.baseline_alpha = None
        self.cov = cov

    def fit(self, X, s, y):
        n, dim = X.shape
        self.baseline_alpha = np.zeros((n, 1))
        y = y.reshape(n, 1)
        s = s.reshape(n, 1) - np.mean(s) * np.ones((n, 1))
        G = (1 / n) * s.T.dot(X)

        def dual_objective(alpha, sign=1.0):
            q = y*X
            Q = q.dot(q.T)
            p = np.ones((1, n))
            return -p.dot(alpha) + 0.5*alpha.T.dot(Q).dot(alpha)

        def grad_objective(alpha, sign=1.0):
            q = y*X
            Q = q.dot(q.T)
            p = np.ones((1, n))
            return p + alpha.T.dot(Q)

        def cov_const_pos(alpha):
            self.w = (y*X).T.dot(alpha)
            return G.dot(self.w) + self.cov

        def cov_const_neg(alpha):
            self.w = (y*X).T.dot(alpha)
            return self.cov - G.dot(self.w)

        cons = (
            {'type': 'eq',
                 'fun': lambda alpha: y.T.dot(alpha),
                 'jac': lambda alpha: y.reshape(1, n)},
                # {'type': 'ineq',
                #  'fun': lambda alpha: alpha.reshape(n, 1),
                #  'jac': lambda alpha: np.ones((1, n))},
                # {'type': 'ineq',
                #  'fun': lambda alpha: self.svm_c*np.ones((n, 1)) - alpha.reshape(n, 1),
                #  'jac': lambda alpha: -np.ones((1, n))},
                {'type': 'ineq',
                 'fun': cov_const_pos,
                 'jac': lambda alpha: G.dot((y*X).T)},
                {'type': 'ineq',
                 'fun': cov_const_neg,
                 'jac': lambda alpha: -G.dot((y*X).T)})

        res = minimize(dual_objective, self.baseline_alpha, jac=grad_objective,
               constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})

        self.w = res.x

    def predict(self, X):
        return sigmoid(X.dot(self.w))

def cross_val(classifier, run):
    X, s, y = load_data()

    roc = pd.DataFrame()

    cv = KFold(n_splits=10, shuffle=False)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)


    for train, test in cv.split(X):
        X_train, X_test = X[train, :], X[test, :]
        s_train, s_test = s[train], s[test]
        y_train, y_test = y[train], y[test]

        classifier.fit(X_train, s_train, y_train)
        preds_k = classifier.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, preds_k)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0


    roc = pd.DataFrame({'tpr': mean_tpr, 'fpr': mean_fpr})

    roc.to_csv(str(classifier)+str(run)+"_roc.csv", index=False)

    mean_tpr /= 10
    mean_tpr[-1] = 1.0
    auc_score = auc(mean_fpr, mean_tpr)
    print("AUC: " + str(auc_score))

    return (auc_score)


if __name__=="__main__":
    C = 0.1
    for run in range(10):
        c = np.array([C])
        classifier = svm_fairness_const(0.2, c)
        cross_val(classifier, run)
        C += 0.1