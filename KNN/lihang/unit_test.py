# -*-coding:utf-8-*-
from knn import *
import numpy as np


def test_e31():
    X = np.loadtxt("Input/data_3-1.txt")
    # print(X-X[0])
    rst = np.linalg.norm(X - X[0], ord=1, axis=1)
    for p in range(2, 5):
        rst = np.vstack((rst, np.linalg.norm(X - X[0], ord=p, axis=1)))
    # Lp(x1,x2)
    # Lp(x1,x3)
    # print(np.round(rst[:, 2], 2).tolist())


def test_e32():
    X = np.loadtxt("Input/data_3-2.txt")
    clf = KNN()
    clf.fit(X)
    print(clf.kdtree)


def test_e33():
    pass


def test_q31():
    pass


def test_q32():
    X = np.loadtxt("Input/data_3-2.txt")
    target = np.array([3, 4.5])
    clf = KNN()
    clf.fit(X)
    rst = clf.predict(target)
    print(rst)


def test_q33():
    pass


if __name__ == '__main__':
    pass
