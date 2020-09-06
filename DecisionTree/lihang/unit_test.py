#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
from dt import *
import pandas as pd
import numpy as np


def test_e51():
    raw_data = pd.read_csv("./Input/data_5-1.txt")
    print(raw_data)


def test_e52():
    raw_data = pd.read_csv("./Input/data_5-1.txt")
    hd = dt._cal_entropy(raw_data[raw_data.columns[-1]])

    rst = np.zeros(raw_data.columns.shape[0] - 1)
    # note: _gain(ID, y) = ent(y)
    for idx, col in enumerate(raw_data.columns[1:-1]):
        hda = dt._gain(raw_data[col], raw_data[raw_data.columns[-1]])
        print(hda)
        rst[idx] = hda
        # print(idx, col, hda)
    # print(rst)
    # print(np.argmax(rst))
    print(hd)


def test_e53():
    raw_data = pd.read_csv("./Input/data_5-1.txt")
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]
    # default criterion: gain
    clf = dt()
    clf.fit(X, y)
    print("gain")
    rst = {'有自己的房子': {'否': {'有工作': {'否': {'否': None}, '是': {'是': None}}}, '是': {'是': None}}}
    print(clf.tree)


def test_q51():
    raw_data = pd.read_csv("./Input/data_5-1.txt")
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]
    # criterion: gain_ratio
    clf = dt(criterion="gain_ratio")
    clf.fit(X, y)
    print("gain_ratio")
    rst = {'有自己的房子': {'否': {'有工作': {'否': {'否': None}, '是': {'是': None}}}, '是': {'是': None}}}
    print(clf.tree)


def test_e54():
    raw_data = pd.read_csv("./Input/mdata_5-1.txt")
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]
    clf = dt()
    clf.fit(X, y)
    print(clf.tree)


def test_predict(self):
    raw_data = pd.read_csv("./Input/mdata_5-1.txt")
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]

    clf = dt(criterion="gain_ratio")
    clf.fit(X, y)
    rst = clf.predict(X[:1])
    self.assertEqual(rst, y[:1].values)
    print("predict: ", rst, "label: ", y[:1])


def test_pruning(self):
    raw_data = pd.read_csv("./Input/mdata_5-1.txt")
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]

    # pre pruning
    clf = dt(criterion="gain_ratio", min_samples_leaf=4)
    clf.fit(X, y)
    print(clf.tree)
    print(clf.num_leaf)
    clf = dt(criterion="gain_ratio", min_samples_leaf=3)
    clf.fit(X, y)
    print(clf.tree)
    print(clf.num_leaf)


if __name__ == '__main__':
    pass
