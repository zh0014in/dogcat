from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import numpy as np
import pandas as pd
from numpy.core.defchararray import index

train = pd.read_csv('train_out.csv',sep=",")
data_train = np.array(train)
X = data_train[:, 10:]
y = data_train[:, 3]

test = pd.read_csv('test_out.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 8:]
i_test = data_test[:, 0]


i_train = data_train[:5000, 0]
X = data_train[:5000, 10:]
y = data_train[:5000, 3]

print(X)
RANDOM_STATE = 731
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'n_estimators': [900,1000,1100,1200], 'max_features': [None,"log2","sqrt","auto"]}]

scores = ['recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE), tuned_parameters, cv=5,
                       scoring='%s_macro' % score, n_jobs=2)
    clf.fit(X_train, y_train)

    y_test = clf.predict_proba(X_test)
    result = np.column_stack([i_test, y_test])
    thefile = open('result_random_forest.csv', 'w')
    thefile.write("ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\n")
    for item in result:
        thefile.write("%s\n" % ",".join(str(x) for x in item))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()