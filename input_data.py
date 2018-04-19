import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os
from sklearn import preprocessing


def readData(path):
    """
    path 为.csv路径
    :param path:
    :return:
    """
    whole_data = pd.read_csv(path, encoding='utf-8')
    whole_data = whole_data.fillna(0)
    whole_list = np.array(whole_data)
    for i in range(len(whole_list)):
        for temp in range(len(whole_list[i])):
            if whole_list[i][temp] == " ":
                whole_list[i][temp] = 0
    # print("转换前：", whole_list)
    whole_list = np.array(whole_list, dtype="float")
    # print(whole_list)
    return whole_list


def makeData(data, flag, batch_size):
    """
    :param data: readData得到的list
    :param flag: "train"和"test"
    :param batch_size: 批次大小
    :return: 打包好的  x : [ batch_size , size ]和 label : [[batch_size]]
    """

    x = []
    y = []
    ran = []
    # print(len(data))
    # print(len(data[1480]))
    # print(data)
    if flag == "train":
        for i in range(batch_size):
            index = random.randint(0, 6384)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[37])]
            y.append(label)
            row = np.delete(row, 37)
            x.append(row)
            ran.append(index)
    else:
        for i in range(batch_size):
            index = random.randint(0, 66)
            # print("test data len", len(data))
            row = data[index]
            label = [int(row[37])]
            y.append(label)
            row = np.delete(row, 37)
            x.append(row)
            ran.append(index)
    # print(label)
    # print("lenx:",len(x))
    # print("len x1:",len(x[1]))
    # x = tf.reshape(x, shape=[batch_size, size])
    # label = tf.reshape(label, shape=[batch_size, 2])

    return x, y
