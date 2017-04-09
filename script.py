import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
# import plotly.plotly as py

train = pd.read_csv('train.csv',sep=",")
test = pd.read_csv('test.csv',sep=",")
# full = np.vstack([train, test])

data_train = np.array(train)

outcometype_count = Counter(data_train[:, 3])
outcomesubtype_count = Counter(data_train[:, 4])

df = pd.DataFrame(outcomesubtype_count)
plt.figure();
df.ix[5].plot.bar();
plt.axhline(0, color='k')

def plot_bar_from_counter(counter, ax=None):
    """"
    This function creates a bar plot from a counter.

    :param counter: This is a counter object, a dictionary with the item as the key
     and the frequency as the value
    :param ax: an axis of matplotlib
    :return: the axis wit the object in it
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    frequencies = counter.values()
    names = counter.keys()

    x_coordinates = np.arange(len(counter))
    ax.bar(x_coordinates, frequencies, align='center')

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(names))

    return ax


plot_bar_from_counter(outcometype_count)
plt.show()
plot_bar_from_counter(outcomesubtype_count)
plt.show()
# fig = plt.gcf()
#
# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')


# data_train = np.array(train)
#
# i_train = data_train[:, 0]
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