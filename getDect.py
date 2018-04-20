import ioUtils
import numpy as np

data_path = r"E:\python_programes\PingAnGame\data\train.csv"
whole_list = ioUtils.readData(data_path)
kind_64_dict, kind_index, nameList = ioUtils.getNameDict(whole_list)
# print(kind_64_dict)
whole_list = np.array(whole_list)
maker_dicts = {}

# 9列
dict = {}
temp = 0
for row in range(len(whole_list)):
    dict[whole_list[row][9]] = dict.get(whole_list[row][9], -1)
    if dict[whole_list[row][9]] == -1:
        dict[whole_list[row][9]] = temp
        temp += 1
    else:
        print("added :", temp)
maker_dicts[9] = dict

# print(maker_dicts)


# 19列
test = []
dict = {}
temp = 0
for row in range(len(whole_list)):
    dict[whole_list[row][19]] = dict.get(whole_list[row][19], -1)
    if dict[whole_list[row][19]] == -1:
        dict[whole_list[row][19]] = temp
        temp += 1
    else:
        print("added : ", temp)
maker_dicts[19] = dict

temp = 0
dict = {}
for row in range(len(whole_list)):
    dict[whole_list[row][20]] = dict.get(whole_list[row][20], -1)
    if dict[whole_list[row][20]] == -1:
        dict[whole_list[row][20]] = temp
        temp += 1
    else:
        print("added : ", temp)
maker_dicts[20] = dict
print(maker_dicts[20])
ioUtils.pickle_save(maker_dicts, "./data/makerDict.txt")
