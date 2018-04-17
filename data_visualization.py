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
    kind_index = {}

    name_df = wholeList[:0]
    namelist = list(name_df)
    index = 0
    for i in namelist:
        kind_index[index] = i
        index += 1
        kind_64_dict[i] = {}  # 注意python里都是引用，所以这里如果用temp来初始化的话，会出现一改全改的情况
    # print(kind_64_dict)
    return kind_64_dict, kind_index


def getEveryItemAcc(kind_64_dict, wholeList, kind_index):

    # 第47个是label，也就是46index
    add_count = 0
    for row in wholeList:
        col_location = 0
        # print(row)
        # print(kind_index)
        for col in row:
            # print("------", kind_64_dict[kind_index[col_location]], "-----------", kind_index[col_location], "的", [col],
            #       "添加了", int(row[46]), "-------现在是",
            #       kind_64_dict[kind_index[col_location]].get(col, 0) + int(row[46]), "--------------")
            kind_64_dict[kind_index[col_location]][col] = \
                kind_64_dict[kind_index[col_location]].get(col, 0) + int(row[46])

            col_location += 1
            # if col_location == 20:
            #     print(kind_64_dict)
            #     return kind_64_dict
        # col_location = 0
        # return kind_64_dict
        add_count += 1
        print(add_count)
    return kind_64_dict


def getWholeArrayList():
    whole_ist = pd.read_csv(data_path, encoding='ISO-8859-1')
    return whole_ist


def draw(kind_64_dict, name_list):
    # with open("./data/draw.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for name in name_list:
    #         writer.writerow([name])
    #         for key in kind_64_dict[name].keys():
    #             writer.writerow([key, kind_64_dict[name][key]])
    # test_names = ["zip_code", "total_rev_hi_lim", "total_rec_int", "total_bal_il", "tot_cur_bal", "revol_util",
    #               "open_rv_12m","mths_since_rcnt_il", "id", "dti_joint", "addr_state"]
    for name in name_list:
        index = 1
        # sorted(kind_64_dict[name].keys())
        x_ = []
        y_ = []
        # if name in test_names:
        try:
            sorted(kind_64_dict[name].items(), key=lambda e: e[0])
            for key in kind_64_dict[name].keys():
                x_.append(key)
                y_.append(kind_64_dict[name][key])
                # print(key,":",kind_64_dict[name][key])

            plt.subplot(1, 2, index)
            plt.scatter(x_, y_, s=10)
            index += 1
            plt.subplot(1, 2, index)
            plt.xticks(np.arange(len(x_)), x_)
            rects = plt.bar(np.arange(len(x_)), y_, width=0.8)
            plt.title(name)
            plt.show()
        except BaseException:
            print(name, "error")
            # else:
            #     print(name,"is not exist")


if __name__ == "__main__":
    whole_list = getWholeArrayList()
    # whole_list = whole_list.fillna(NAN)
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
    draw(kind_64_dict, name_list)
