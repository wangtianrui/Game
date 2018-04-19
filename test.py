import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
import tensorflow as tf
from sklearn.model_selection import cross_val_score
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

help(cross_val_score)