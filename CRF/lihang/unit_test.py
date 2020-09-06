#! /usr/bin/env python
#! -*- coding=utf-8 -*-
from crf import *
from sympy import *
import numpy as np
import argparse
import logging
import unittest
import re


def test_e111():
    # Q:针对11.1这个问题, 为什么要求非规范话的条件概率?
    # A:参考下书中预测算法部分
    Y = np.array([1, 2, 2])
    # 5 + 4 = 9
    w_k = np.array([1, 0.6, 1, 1, 0.2, 1, 0.5, 0.8, 0.5])
    # todo: 这里再思考下转移特征和状态特征的构建
    # todo: 注意联系最后的公式， 思考构建一个特征函数需要的参数与格式
    f_k = np.zeros(9)
    # transition feature
    # i-1, i
    f_k[0] = np.sum([1 if tmp[0] == 1 and tmp[1] == 2 else 0 for tmp in list(zip(Y[:-1], Y[1:]))])
    f_k[1] = np.sum([1 if tmp[0] == 1 and tmp[1] == 1 else 0 for tmp in list(zip(Y[:-1], Y[1:]))])
    f_k[2] = np.sum([1 if tmp[0] == 2 and tmp[1] == 1 else 0 for tmp in list(zip(Y[:-1], Y[1:]))])
    f_k[3] = np.sum([1 if tmp[0] == 2 and tmp[1] == 1 else 0 for tmp in list(zip(Y[:-1], Y[1:]))])
    f_k[4] = np.sum([1 if tmp[0] == 2 and tmp[1] == 2 else 0 for tmp in list(zip(Y[:-1], Y[1:]))])
    # state feature
    # i
    f_k[5] = np.sum([1 if tmp == 1 else 0 for tmp in [Y[0]]])
    f_k[6] = np.sum([1 if tmp == 2 else 0 for tmp in Y[:2]])
    f_k[7] = np.sum([1 if tmp == 1 else 0 for tmp in Y[1:]])
    f_k[8] = np.sum([1 if tmp == 2 else 0 for tmp in [Y[2]]])

    # 生成全局特征向量
    proba = np.sum(w_k*f_k)
    print("P(y|x) proportional to exp(%1.1f)" % proba)

def test_e112():
    a01, a02, b11, b12, b21, b22, c11, c12, c21, c22 = symbols("a01, a02, b11, b12, b21, \
                                                                b22, c11, c12, c21, c22")
    M1 = Matrix([[a01, a02],
                 [0,   0]])
    M2 = Matrix([[b11, b12],
                 [b21, b22]])

    M3 = Matrix([[c11, c12],
                 [c21, c22]])

    M4 = Matrix([[1, 0],
                 [1, 0]])
    Z = (M1 * M2 * M3 * M4)[0].expand()
    P = str(Z).replace(" ", "").split("+")
    # 体会各个路径之间关系
    for i in range(2):
        for j in range(2):
            for k in range(2):
                print(str(M1[0, i] * M2[i, j] * M3[j, k]))
    print(Z)
    print(P)


def test_e113():
    pass

def test_readtemplate():
    # 要清楚特征函数模板和特征函数不是一个概念
    tpl = "\[(.*?)\]"
    regex = re.compile(tpl)

    with open("./Input/template") as f:
        for line in f:
            if line[0] == 'U':
                raw = line.strip().split(":")
                print(raw[0])
                print(regex.findall(raw[1]))


if __name__ == '__main__':
    test_e111()
