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


# a = np.array([[1,2],[2,3],[4,5]])
# b = random.sample(list(a),2)
# c = np.delete(b,0,axis=1)
# print(b)
# print(c)
import pickle


def pickle_save(maker_dicts):
    """
    保存maker字典
    :return:
    """
    f = open("./data/123456test.txt", "wb")
    pickle.dump(maker_dicts, f)
    f.close()


def pickle_load():
    """
    读取字典
    :return:
    """
    f = open("./data/123456test.txt", "rb")
    maker_dicts = pickle.load(f)
    return maker_dicts


a = [59793399.0, 18900.0, 18900.0, 18900.0, 36.0, 14.65, 651.94000000000005, 500.0, 630.0, 13.0, 110.0, 443557.0,
     75000.0, 329558.0, 201508.0, 6253.0, 0.0, 524215.0, 331275.0, 1374.0, 26742.0, 23.629999999999999, 1989080.0,
     73.409498846159437, 0.0, 7259.0, 69.799999999999997, 30.0, 0.0, 17184.09, 17184.09, 2592.3800000000001,
     2592.3800000000001, 1715.9100000000001, 876.47000000000003, 0.0, 0.0, 0.0, 0.0, 19.0, 1.0, 886868.0,
     120692.69378428547, 14.275468103405164, 220.48855716848189, 0.0, 72472.0, 1.2126667337263368, 2.7317062299399519,
     0.94773126558780241, 1.8065457843337036, 21.965550246466783, 32798.936774481088, 67.132657630134688,
     1.3504149015366862, 2.5801443101304327, 5584.2418002602462, 70.858445462225845, 10400.0, 1.0965788379925343,
     1.7112021985994963, 2.0925150230755363]
index = 0
for i in a:
    if i == 0.0:
        print(i)
    else:
        index += 1
print(a)
print(b)
