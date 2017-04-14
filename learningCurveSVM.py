from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import numpy as np
import pandas as pd

train = pd.read_csv('train_out.csv',sep=",")
data_train = np.array(train)
X = data_train[:, 10:]
y = data_train[:, 3]

test = pd.read_csv('test_out.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 8:]
i_test = data_test[:, 0]

c = 300.0
g = 0.001
k = 'rbf'
clf = SVC(C=c, gamma=g, kernel=k,probability = True, cache_size=4000)

train_sizes, train_scores, valid_scores = learning_curve(
    clf, X, y, train_sizes=[2000, 4000, 6000, 8000, 10000], cv=5, n_jobs=5)
print train_sizes

print train_scores



print valid_scores
