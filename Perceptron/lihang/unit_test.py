# -*-coding:utf-8-*-
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from perceptron import *
import numpy as np


def test_e21():
    print("test case e21")
    # data e2.1
    data_raw = np.loadtxt("Input/data_2-1.txt")
    X = data_raw[:, :2]
    y = data_raw[:, -1]
    clf = Perceptron(eta=1, verbose=False)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(clf.w)
    print(str(y_pred))


def test_e22():
    print("test case e22")
    # data e2.1
    data_raw = np.loadtxt("Input/data_2-1.txt")
    X = data_raw[:, :2]
    y = data_raw[:, -1]
    clf = Perceptron(verbose=False)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(clf.w)
    print(str(y_pred))


def test_logic_1():
    # loaddata
    data_raw = np.loadtxt("Input/logic_data_1.txt")
    X = data_raw[:, :2]
    clf = Perceptron(max_iter=100, eta=0.0001, verbose=False)
    # test and
    y = data_raw[:, 2]
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("test case logic_1 and")
    # test or
    print("test logic_1 or")
    y = data_raw[:, 3]
    clf.fit(X, y)
    y_pred = clf.predict(X)
    # test not
    print("test logic_1 not")
    y = data_raw[:, 4]
    clf.fit(X, y)
    y_pred = clf.predict(X)


def test_logic_2():
    # loaddata
    data_raw = np.loadtxt("Input/logic_data_2.txt")
    X = data_raw[:, :3]
    clf = Perceptron(max_iter=100, eta=0.0001, verbose=False)
    # test and
    y = data_raw[:, 3]
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("test case logic_2 and")
    # test or
    print("test logic_2 or")
    y = data_raw[:, 4]
    clf.fit(X, y)
    y_pred = clf.predict(X)
    # test not
    print("test logic_2 not")
    y = data_raw[:, 5]
    clf.fit(X, y)
    y_pred = clf.predict(X)


def test_mnist():
    raw_data = load_digits(n_class=2)
    X = raw_data.data
    y = raw_data.target
    # 0和1比较容易分辨吧
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)

    clf = Perceptron(verbose=False)
    clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    score = accuracy_score(y_test, test_predict)
    print("The accruacy socre is %2.2f" % score)


if __name__ == '__main__':
    pass
