#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
import numpy as np


class TripleCoin(object):
    def __init__(self, pi=0, p=0, q=0):
        self.pi = pi
        self.p = p
        self.q = q

    def sample(self,
               n=10):
        """
        e9.1, 三硬币模型数据
        :param n:
        :return:
        """
        rst = np.empty(1)
        for n_iter in range(n):
            pi_ = np.random.binomial(1, self.pi, 1)
            if pi_:
                rst = np.hstack((rst, np.random.binomial(1, self.p, 1)))
            else:
                rst = np.hstack((rst, np.random.binomial(1, self.q, 1)))
        return rst[1:]
