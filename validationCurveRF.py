import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
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

clf = GradientBoostingClassifier( max_features=None) 
train_scores, valid_scores = validation_curve(clf, X, y, "max_depth", [3,4,5])
print train_scores

print valid_scores
