import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
from numpy.linalg import cholesky
import random
import math
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

#
# sampleNo = 1000
#
#
# # 一维正态分布
# # 下面三种方式是等效的
# def sig(x):
#     y = tf.nn.sigmoid(x)
#     print(y)
#     return y
#
#
# f = tf.placeholder(dtype="float")
# y = sig(f)
#
# with tf.Session() as sess:
#     array = []
#     temp = 1
#     x_star = 41
#     x_ = []
#     # s = np.random.normal(mu, sigma, sampleNo)
#     for i in range(-100, 100):
#         f = float(i)
#         # print(y)
#
#         array.append(sess.run(y,feed_dict={f:i}))
#         # x_.append(i)
#         x_.append(x_star)
#         x_star += temp
#
#     # for i in range(8):
#     #
#     #     index += 1
#     plt.subplot(1, 1, 1)
#     plt.xticks(np.arange(len(x_)), x_)
#     rects = plt.bar(np.arange(len(x_)), array, width=0.9)
#     plt.show()


#
#
#
# def setPict(x_, y_):
#     array = y_
#     plt.subplot(1, 1, 1)
#     plt.xticks(np.arange(len(x_)), x_)
#     rects = plt.bar(np.arange(len(x_)), array, width=0.9)
#
#     for index in range(len(rects)):
#         if index <= 15:
#             rects[index].set_color("green")
#         elif index > 15 and index < 25:
#             rects[index].set_color("blue")
#         else:
#             rects[index].set_color("red")
#
#     plt.grid(True)
#     plt.text(4, 10, "x：环境声音分贝值\ny：用户耳机声音适应分贝值", size=15)
#     plt.title("the class four")
#
#
# # setPict(x_, array)
# # plt.show()
#
# c = csv.writer(open("./bao.csv", "w", newline=""))
# for t in range(50):
#     array = []
#     temp = 1
#     x_star = 41
#     x_ = []
#
#     def sig(x):
#         y = float(1) / (1 + math.exp(-x))
#         rand = float(random.randint(0, t)) / 1000
#         y += rand
#         return round((y + 0.05) * 12 + 1,2)
#
#     temp = 40
#
#     for i in range(-100, 100):
#         if i % 5 == 0:
#             x_.append(temp)
#             temp += 1
#             print(temp, "  ", i)
#             array.append(sig(i / 15))
#     for i in range(len(x_)):
#         c.writerow([x_[i], array[i]])
#     c.writerow([" "])


a = [[1,2],[2,3],[4,5]]
b = [item[1] for item in a]
print(b)
