import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.ensemble import GradientBoostingClassifier

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

clf = GradientBoostingClassifier(max_depth=5, max_features=None)


clf.set_params(n_estimators=900)
clf.fit(X, y)
print clf.feature_importances_

# Generate the "OOB error rate" vs. "n_estimators" plot.
# for label, clf_err in error_rate.items():
#     xs, ys = zip(*clf_err)
#     plt.plot(xs, ys, label=label)
#
# plt.xlim(min_estimators, max_estimators)
# plt.xlabel("n_estimators")
# plt.ylabel("OOB error rate")
# plt.legend(loc="upper right")
# plt.show()

y_test = clf.predict_proba(X_test)
result = np.column_stack([i_test, y_test])
thefile = open('GBRTresult.csv', 'w')
thefile.write("ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\n")
for item in result:
    thefile.write("%s\n" % ",".join(str(x) for x in item))
