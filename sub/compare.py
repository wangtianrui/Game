import pandas as pd
import numpy as np

lihao_sub = pd.read_csv("lihaosub.csv")
wang_sub = pd.read_csv("wangsub.csv")

wang_labels = wang_sub["acc_now_delinq"]
lihao_labels = lihao_sub["acc_now_delinq"]
# print(wang_labels)
# print(lihao_labels)
wanglist = np.array(wang_labels)
lihao_labels = np.array(lihao_labels)

compare = (wanglist == lihao_labels)
same = 0
same1 = 0
wang1 = 0
lihao1 = 0
for i in range(len(compare)):
    if compare[i] == True and wanglist[i] == 1:
        same1 += 1
    if compare[i] == True:
        same += 1

print("1相同的个数:", same1)
print("相同的总个数：", same)
print(len(compare))

