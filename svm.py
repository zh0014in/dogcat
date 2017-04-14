import numpy as np
import pandas as pd
from sklearn import svm

train = pd.read_csv('train_out.csv',sep=",")
data_train = np.array(train)
X = data_train[:, 10:15]
y = data_train[:, 3]

test = pd.read_csv('test_out.csv', sep=",")
data_test = np.array(test)
X_test = data_test[:, 8:13]
i_test = data_test[:, 0]

print X

c = 300.0
g = 0.001
k = 'rbf'
clf = svm.SVC(C=c, gamma=g, kernel=k,probability = True)

print clf.fit(X, y)

result = clf.predict_proba(X_test)

result = np.column_stack([i_test, result])
print result
thefile = open('result_svm.csv', 'w')
thefile.write("ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\n")
for item in result:
    thefile.write("%s\n" % ",".join(str(x) for x in item))