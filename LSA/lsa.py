#! /usr/bin/env python
# 截断奇异值分解用在count/tf-idf矩阵的时候，叫做潜在语义分析。
import numpy as np


class LSA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.singular_values = None
        self.explained_variance_ratio = None
        self.u = None

    def fit(self, x):
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        max_abs_raws = np.argmax(np.abs(vh), axis=1)
        signs = np.sign(vh[range(vh.shape[0]), max_abs_raws])
        u *= signs
        vh *= signs[:, np.newaxis]
        k = self.n_components
        self.components = vh[:k]
        self.singular_values = s[:k]
        x_transformed = u * s
        self.explained_variance = np.var(x_transformed, axis=0)
        self.explained_variance_ratio = (self.explained_variance / self.explained_variance.sum())[:k]
        self.explained_variance = self.explained_variance[:k]
        self.u = u
        return x_transformed
