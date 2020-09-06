# coding:utf-8
import numpy as np
from numpy.random import normal, random, uniform
import pickle
import tensorflow as tf
import matplotlib.pylab as plt
import math
import gym
import time
import torch

env = gym.make('Tsp-Mini-v1')
env.reset()
random_episodes = 0
reward_sum = 0
action = 0
n = 6
random_city = np.random.permutation(range(n))
print(random_city)

# 进行10次随机试验
while random_episodes < 3:
    observation, reward, done, _ = env.step(random_city[action])
    action += 1
    reward_sum += reward
    env.render()
    if done:
        random_city = np.random.permutation(range(n))
        print(random_city)
        print("Reward for this episode was: ", reward_sum)
        reward_sum = 0
        random_episodes += 1
        action = 0
        env.reset()

# 定义超参数
H = 100
batch_size = 1
learning_rate = 0.01
gamma = 0.99
D = n

# 定义策略网络的具体结构
tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, D],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.softmax(tf.nn.relu(score))

# 定义策略网络的loss函数
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, n], name="input_y")
advantages = tf.placeholder(tf.float32, [None, 1], name="reward_signal")
cross_entropy = tf.reduce_mean(tf.reduce_sum(input_y * tf.log(probability)))
loss = -tf.reduce_mean(cross_entropy * advantages)
newGrads = tf.gradients(loss, tvars)

# 积累梯度并更新参数
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


# 计算每个Action的潜在期望
def discount_rewards(r):
    discount_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r


# 定义参数
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 0
total_episodes = 100
init = tf.initialize_all_variables()

# 模型开始工作
with tf.Session() as sess:
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    while episode_number < total_episodes:
        # 进行一次试验
        x = np.reshape(observation, [1, D])
        xs.append(x)

        tfprob = sess.run(probability, feed_dict={observations: x}).reshape(n)
        cum_tfprob = np.cumsum(tfprob)
        ran = np.random.uniform()
        for k in range(len(cum_tfprob)):
            if k ==0:
                if cum_tfprob[k] >= ran:
                    action = 0
            else:
                if (cum_tfprob[k] >= ran) and (ran > cum_tfprob[k - 1]):
                    action = k

        y = np.zeros(n).astype(int)
        y[action] = 1
        ys.append(y)

        observation, reward, done, info = env.step(action)
        env.render()
        reward_sum += reward
        drs.append(reward)

        # time.sleep(0.5)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            visited = []
            xs, ys, drs = [], [], []
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            # 迭代次数达到batch_size
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                print('Average reward for episode %f.' % (reward_sum / batch_size))
                reward_sum = 0
            observation = env.reset()
