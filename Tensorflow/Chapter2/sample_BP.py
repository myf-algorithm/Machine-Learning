# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState


if __name__ == '__main__':
    batch_size = 8
    w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))  # 隐含层 3个节点
    w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input") # 输入层两个节点
    y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input') # 输出层一个节点

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) # 没有采用激活函数
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    rdm = RandomState(1)
    X = rdm.rand(128,2)
    Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 输出目前（未经训练）的参数取值。
        print("w1:", sess.run(w1))
        print("w2:", sess.run(w2))
        print("\n")

        # 训练模型。
        STEPS = 5000
        for i in range(STEPS):  # 自己分批，每批8个样本，一共128个样本
            start = (i * batch_size) % 128
            end = (i * batch_size) % 128 + batch_size
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 == 0:  # 运行1000次就打印下代价函数值，方便观察
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

        # 输出训练后的参数取值。
        print("\n")
        print("w1:", sess.run(w1))
        print("w2:", sess.run(w2))
