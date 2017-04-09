import numpy as np
import pandas as pd

train = pd.read_csv('train.csv',sep=",")
test = pd.read_csv('test.csv',sep=",")
# full = np.vstack([train, test])

data_train = np.array(train)

i_train = data_train[:, 0]
age_train = data_train[:, 7]
age_train_time_value = np.split(age_train, 1);
print age_train_time_value
# X_train = data_train[:, 2:]
# y_train = data_train[:, 1]
#
# #find hidden feature: same word should have something the same
# position = X_train[:, 1]
# X_train[:, 0] = abs(i_train - position)/100
#
# #remove features with low variance
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(X_train)
# print X_train
#
# p_train = np.column_stack([i_train, y_train])
#
# from sklearn import svm
#
# test = pd.read_csv('test.csv', sep=",")
# data_test = np.array(test)
# X_test = data_test[:, 2:]
# i_test = data_test[:, 0]
#
# position = X_test[:, 1]
# X_test[:,0] = abs(i_test - position)/100
# #remove features with low variance
# from sklearn.feature_selection import VarianceThreshold
# sel.fit_transform(X_test)
#
# #use GridSearchCV to find this
# c = 500.0
# g = 0.01
# k = 'rbf'
# clf = svm.SVC(C=c, gamma=g, kernel=k)
#
# print clf.fit(X_train,y_train)
#
# result = clf.predict(X_test)
#
# result = np.column_stack([i_test, result])
# result = np.vstack([p_train, result])
# print result
# thefile = open('result.csv', 'w')
# thefile.write("Id,Prediction\n")
# for index, item in enumerate(result):
#     thefile.write("%s" % item[0])
#     thefile.write(",%s\n" % item[1])