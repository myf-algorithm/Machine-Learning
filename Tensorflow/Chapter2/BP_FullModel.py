# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 500  # 隐藏层数

BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  # 学习率初始化
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
REGULARAZTION_RATE = 0.0001  # 正则化参数
TRAINING_STEPS = 5000  # 训练次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减系数


# 前向
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类--本质就是在每次计算时候，使用了历史值，然后取平均，防止某一次参数变化太快
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # 截断正态分布，防止饱和
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 训练时候 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)  # 不可训练的变量
    # 让越近的数据权重越大，越远的数据权重越小 MOVING_AVERAGE_DECAY控制 并采用num_updates 参数来动态设置 decay 的大小
    # shadow_variable=decay×shadow_variable+(1−decay)×variable
    # variable 当前实际值，=后面的shadow_variable 前面运算保存的影子值
    # 可以看出：decay越大，考虑历史值就越多，模型越稳定 = 指数衰减滑动平均模型  shadow_variable 运算后实际值，用于替换variable
    # 在刚开始训练时候，需要把decay调小，模型更新速度加快，训练后期，decay要变大，模型更新速度变慢，通过global_step=num_updates控制
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # 指数衰减滑动平均模型
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 必须手动调用该函数，滑动一下
    # 验证时候使用滑动指数平均
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    # http://blog.csdn.net/QW_sunny/article/details/68960838
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # label必须是索引
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算 L2范数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。本质为：
    # learning_rate= LEARNING_RATE_BASE *LEARNING_RATE_DECAY*exp(global_step/(mnist.train.num_examples / BATCH_SIZE))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 初始学习率
        global_step,  # 变量 ，初始是0
        mnist.train.num_examples / BATCH_SIZE,  # 全部训练完一次，需要迭代多少次(衰减步长，即每跑多少次才衰减)
        LEARNING_RATE_DECAY,  # 学习率衰减系数
        staircase=True)  # true 阶梯型衰减(每训练步长次数，就衰减LEARNING_RATE_DECAY)  false 连续型衰减(每一次训练都衰减)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                           global_step=global_step)  # 不要忘记了 global_step

    # 反向传播更新参数和更新每一个参数的滑动平均值
    # 控制计算流图，给图中的某些计算指定顺序
    # http://blog.csdn.net/NNNNNNNNNNNNY/article/details/70177509

    # 等价 train_op = tf.group([train_step, variables_averages_op]) 一次运算，全部操作完成
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')  # 这行代码什么也不做，只是构造了一次操作组而已

    # 计算正确率--验证时候使用
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


def main(argv=None):
    mnist = input_data.read_data_sets("../dataSet/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
