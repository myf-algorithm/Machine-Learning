# -*-coding:utf-8-*-
from nb import *
import pandas as pd


def test_e41():
    data = pd.read_csv("./Input/data_4-1.txt", header=None, sep=",")
    X = data[data.columns[0:2]]
    y = data[data.columns[2]]
    clf = NB(1)
    clf.fit(X, y)
    rst = clf.predict([2, "S"])


def test_e42():
    data = pd.read_csv("./Input/data_4-1.txt", header=None, sep=",")
    X = data[data.columns[0:2]].values
    y = data[data.columns[2]]
    clf = NB(1)
    clf.fit(X, y)
    rst = clf.predict([2, "S"])


if __name__ == '__main__':
    pass
