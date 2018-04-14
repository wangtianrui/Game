import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NAN = 99999999.0
data_path = r"./data/11data.csv"
"""
csv_reader = csv.reader(open(data_path, encoding='utf-8'))
whole_list = []
column_count = len(list(csv_reader)[0])
print(column_count)
column_data = []

for i in csv_reader:
    print(i)
    whole_list.append(i)


for i in range(column_count):
    one_kind = whole_list[:, int(i)]
    column_data.append(one_kind)
    print(one_kind)
"""


def getNameDict(wholeList):
    """
    获取第一行信息
    :param wholeList:
    :return:
    """
    kind_64_dict = {}
    every_item = {}
    kind_index = {}

    name_df = wholeList[:0]
    namelist = list(name_df)
    index = 0
    for i in namelist:
        kind_index[index] = i
        index += 1
        kind_64_dict[i] = every_item
    # print(kind_64_dict)
    return kind_64_dict, kind_index


def getEveryItemAcc(kind_64_dict, wholeList, kind_index):
    # 第47个是label，也就是46index
    for row in wholeList:
        col_location = 0
        for col in row:
            print("-----------------",kind_index[col_location],"的",[col],"添加了",int(row[46]),"---------------------")
            kind_64_dict[kind_index[col_location]][col] = \
                kind_64_dict[kind_index[col_location]].get(col, 0) + int(row[46])
            col_location += 1
    print(kind_64_dict)
    return kind_64_dict


def getWholeArrayList():
    whole_ist = pd.read_csv(data_path)
    return whole_ist


def draw(kind_64_dict, name_list):
    # with open("./data/draw.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for name in name_list:
    #         writer.writerow([name])
    #         for key in kind_64_dict[name].keys():
    #             writer.writerow([key, kind_64_dict[name][key]])
    index = 1
    for name in name_list:
        x_ = []
        y_ = []
        # sorted(kind_64_dict[name].keys())

        for key in kind_64_dict[name].keys():
            x_.append(key)
            y_.append(kind_64_dict[name][key])
            print(key)
            print(name)
        plt.subplot(1000, 1000, index)
        plt.plot(x_, y_)
        plt.title(name)
        index += 1
        print("------------------------------------------------------------",index)
        plt.show()


if __name__ == "__main__":
    whole_list = getWholeArrayList()
    whole_list = whole_list.fillna(NAN)
    # print(whole_list)
    kind_64_dict, kind_index = getNameDict(whole_list)
    # print(kind_index)
    whole_list = np.array(whole_list)

    # print(whole_list)
    kind_64_dict = getEveryItemAcc(kind_64_dict, whole_list, kind_index)

    # print(kind_64_dict)
    name_keys = kind_64_dict.keys()
    # print(name_keys)
    name_list = list(name_keys)
    # print(name_list)
    #draw(kind_64_dict, name_list)
