# -*- coding: utf-8 -*-

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    # Import data
    mnist = input_data.read_data_sets("../dataSet/MNIST_data/", one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W_2 = tf.get_variable(
        'W_2', [784, 30], initializer=tf.random_normal_initializer(stddev=1 / np.sqrt(784.0)))  # 均值为0 方差为1/特征维数的高斯分布
    b_2 = tf.get_variable(
        'b_2', [30], initializer=tf.random_normal_initializer())
    z_2 = tf.matmul(x, W_2) + b_2
    a_2 = tf.sigmoid(z_2)
    W_3 = tf.get_variable(
        'W_3', [30, 10], initializer=tf.random_normal_initializer(stddev=1 / np.sqrt(30.0)))
    b_3 = tf.get_variable(
        'b_3', [10], initializer=tf.random_normal_initializer())
    z_3 = tf.matmul(a_2, W_3) + b_3
    a_3 = tf.sigmoid(z_3)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 第一种代价函数
    # loss = tf.losses.tf.losses.mean_squared_error(y_, a_3)
    # loss = tf.reduce_mean(tf.norm(y_ - a_3, axis=1)**2) / 2   # axis=1 表示对行进行求欧式距离
    # 第二种代价函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3))  # 交叉熵+sigmod
    train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    best = 0
    for epoch in range(30):
        for _ in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(a_3, 1), tf.argmax(y_, 1))  # 返回每一行的最大值索引
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
        accuracy_currut = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels})
        print("Epoch %s: %s / 10000" % (epoch, accuracy_currut))
        best = (best, accuracy_currut)[best <= accuracy_currut]  # 返回两种的最大值

    # Test trained model
    print("best: %s / 10000" % best)


if __name__ == '__main__':
    tf.app.run(main=main)
