#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
from sklearn.datasets import load_digits
from gmm import *
from bmm import *
from model import *
import numpy as np

def test_e91():
    # 这个是个伯努利分布, 例子讲的是EM算法, 不是GMM. 理解这里的关系
    sample = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    print("sample %s" % sample)
    pi = 0.5
    p = 0.5
    q = 0.5
    print("init prob pi=%1.1f, p=%1.1f, q=%1.1f" % (pi, p, q))
    mu = np.ones(sample.shape) / 2
    print(("mu: %s" % mu))
    for n_iter in range(10):
        # E Step
        for j, yj in enumerate(sample):
            if yj:
                mu[j] = pi * p / (pi * p + (1 - pi) * q)
            else:
                mu[j] = pi * (1 - p) / (pi * (1 - p) + (1 - pi) * (1 - q))
        # print(("%d mu: %s" % (n_iter, mu)))
        # M Step
        pi = np.mean(mu)
        p = np.sum(mu * sample) / np.sum(mu)
        q = np.sum((1 - mu) * sample) / np.sum(1 - mu)
        print((n_iter, pi, p, q))

    pi = 0.4
    p = 0.6
    q = 0.7
    print("init prob pi=%1.1f, p=%1.1f, q=%1.1f" % (pi, p, q))
    mu = np.ones(sample.shape) / 2
    print(("mu: %s" % mu))
    for n_iter in range(10):
        # E Step
        for j, yj in enumerate(sample):
            if yj:
                mu[j] = pi * p / (pi * p + (1 - pi) * q)
            else:
                mu[j] = pi * (1 - p) / (pi * (1 - p) + (1 - pi) * (1 - q))
        # print(("%d mu: %s" % (n_iter, mu)))
        # M Step
        pi = np.mean(mu)
        p = np.sum(mu * sample) / np.sum(mu)
        q = np.sum((1 - mu) * sample) / np.sum(1 - mu)
        print((n_iter, pi, p, q))

def test_t93():
    pass

def test_t91():
    # 可以通过TripleCoin来实现采样
    # tc = TripleCoin(pi=0.3, p=0.6, q=0.2)
    # sample = tc.sample()
    # 对比说明同分布的不同序列的参数估计
    sample = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    sample = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    print(sample)
    pi = 0.5
    p = 0.5
    q = 0.5
    # mu = sample*pi
    # mu += (1-sample)*(1-pi)
    mu = np.ones(sample.shape) * 0.5
    print(("mu: %s" % mu))
    for n_iter in range(10):
        for j, yj in enumerate(sample):
            if yj:
                mu[j] = pi * p / (pi * p + (1 - pi) * q)
            else:
                mu[j] = pi * (1 - p) / (pi * (1 - p) + (1 - pi) * (1 - q))
        # print(("%d mu: %s" % (n_iter, mu)))
        pi = np.mean(mu)
        p = np.sum(mu * sample) / np.sum(mu)
        q = np.sum((1 - mu) * sample) / np.sum(1 - mu)
        print((n_iter, pi, p, q))

def test_simulation():
    # 使用高斯生成数据, 然后用gmm拿到模型参数, 对比生成参数与学习到的参数.
    pass

def test_bmm():
    # load data to X
    X = ((load_digits().data[:100]) > 0.5) * 1
    X = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    bmm = BMM(n_components=2, max_iter=2)
    bmm.fit(X)
    print("----bmm 0.5 0.5 0.5----start----")
    print(bmm.alpha[1])
    print(bmm.mu[:, 1])
    print("----bmm----end----")
    bmm = BMM(n_components=2, max_iter=2, alpha=0.4, mu=np.array([0.7, 0.6]))
    bmm.fit(X)
    print("----bmm 0.4 0.6 0.7----start----")
    print(bmm.alpha[1])
    print(bmm.mu[:, 1])
    print("----bmm----end----")


if __name__ == '__main__':
    test_bmm()

