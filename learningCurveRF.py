from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

RANDOM_STATE = 731

train = pd.read_csv('train_out.csv',sep=",")
data_train = np.array(train)
X = data_train[:, 10:]
y = data_train[:, 3]

test = pd.read_csv('test_out.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 8:]
i_test = data_test[:, 0]

estimator = 900

clf = RandomForestClassifier(warm_start=True, max_features=None, oob_score=True, random_state=RANDOM_STATE)

train_sizes, train_scores, valid_scores = learning_curve(
    clf, X, y, train_sizes=[5000, 10000, 15000, 20000], cv=5)
print train_sizes

print train_scores



print valid_scores
