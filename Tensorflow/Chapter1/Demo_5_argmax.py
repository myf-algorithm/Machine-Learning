# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6]])
    tf.InteractiveSession()
    print(tf.argmax(a).eval())  # 返回每一列的最大值索引
    print(tf.argmax(a, 0).eval())   # 返回每一列的最大值索引
    print(tf.argmax(a, 1).eval())    # 返回每一行的最大值索引
    print(np.argmax(a))    # 返回整个array的最大值索引
    print(np.argmax(a, 0))    # 返回每一列的最大值索引
    print(np.argmax(a, 1))    # 返回每一行的最大值索引