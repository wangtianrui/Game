import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.linalg import cholesky
#
# x_ = ["wind", "cloud", "tur", "1", "sd"]
# y_ = [1, 2, 3, 4, 6]
# if 1 in y_:
#     print("ok")
#
# a = np.array([[1, 2, 3, "4213.2", 5, " "], [1, "stinr", 3, "4213.2", 5]])
# for i in range(len(a)):
#     for t in range(len(a[i])):
#         if a[i][t] == " ":
#             a[i][t] = 0
# print(a)

import ioUtils

# ones_like_actuals = tf.ones_like([[0],[1],[1],[1],[0]])
# zeros_like_actuals = tf.zeros_like([[0],[0],[1],[1],[0]])
# sess = tf.Session()
# print(sess.run(zeros_like_actuals))
# print(sess.run(ones_like_actuals))
# predictions = tf.argmax([[0], [1], [1], [1], [0]], 1)
# actuals = tf.argmax([[0], [0], [1], [1], [0]], 1)
# predictions  = tf.Variable([[0], [1], [1], [1], [0]])
# actuals = tf.Variable([[0], [0], [1], [1], [0]])
# print(predictions)
# ones_like_actuals = tf.ones_like(actuals)
# zeros_like_actuals = tf.zeros_like(actuals)
# ones_like_predictions = tf.ones_like(predictions)
# zeros_like_predictions = tf.zeros_like(predictions)
#
# tp_op = tf.reduce_sum(
#     tf.cast(
#         tf.logical_and(
#             tf.equal(actuals, ones_like_actuals),
#             tf.equal(predictions, ones_like_predictions)
#         ),
#         "float"
#     )
# )
#
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(tp_op))
# print(sess.run(tf.equal(actuals, ones_like_actuals)))


import os

# array = np.random.normal(size=(50))
# array = np.abs(array)
# print(array)
# x=[]
# for i in range(50):
#    x.append(i)
# plt.bar(range(len(array)), array)
# plt.show()


sampleNo = 1000
# 一维正态分布
# 下面三种方式是等效的
mu = 3
sigma = 0.1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)
plt.subplot(141)
plt.hist(s, 30, normed=True)

np.random.seed(0)
s = sigma * np.random.randn(sampleNo) + mu
plt.subplot(142)
plt.hist(s, 30, normed=True)

np.random.seed(0)
s = sigma * np.random.standard_normal(sampleNo) + mu
plt.subplot(143)
plt.hist(s, 30, normed=True)


plt.show()