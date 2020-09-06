import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, noise=20, n_features=1)
plt.scatter(X, y)


def least_square():
    # 最小二乘法是一种求解释一种基于均方误差来求解的一种方法，
    # 在线性回归中，最小二乘法就是试图找到一条直线，
    # 使所有的样本到直线上的欧式距离最小
    ones = np.ones(X.shape[0])
    A = np.insert(X, 0, 1, axis=1)  # 特征
    b = y.reshape(-1, 1)
    plt.scatter(X, y)
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    y_pred = A @ x
    plt.plot(X, y_pred, 'r--')
    plt.show()


class LinearRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y = y.reshape(-1, 1)
        self.w = np.random.randn(X.shape[1], 1)
        for _ in range(50):
            y_pred = X @ self.w  # 100 * 1
            mse = np.mean(0.5 * (y_pred - y) ** 2)
            grad_w = X.T @ (y_pred - y)
            self.w -= 0.01 * grad_w
            print(_, mse, self.w[0][0], self.w[1][0])

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.w


def gd():
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    plt.scatter(X, y)
    plt.plot(X, y_pred, 'r--')
    plt.show()


def closed_form():
    x_mean = np.mean(X)
    Y = y.reshape(-1, 1)
    w = np.sum(Y * (X - x_mean)) / (np.sum(X ** 2) - 1. / X.shape[0] * (np.sum(X)) ** 2)
    b = np.sum(Y - w * X) / X.shape[0]
    print(w, b)


if __name__ == '__main__':
    least_square()
    gd()
    closed_form()
