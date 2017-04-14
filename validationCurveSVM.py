import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
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
clf = SVC(C=c, gamma=g, kernel=k,probability = True,cache_size=4000)
train_scores, valid_scores = validation_curve(clf, X, y, "gamma",
                                              np.logspace(-7, 0, 5))
print train_scores

print valid_scores