import pandas as pd
import numpy as np


def readData(datapath):
    whole_ist = pd.read_csv(datapath, encoding='ISO-8859-1')
    whole_ist = whole_ist.fillna(" ")
    return whole_ist


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


if __name__ == "__main__":
    data_path = r"E:\python_programes\PingAnGame\data\11data.csv"
    whole_list = readData(data_path)
    kind_64_dict, kind_index = getNameDict(whole_list)
    print(kind_64_dict)
    whole_list = np.array(whole_list)
    database = []
    # 前三列
    for row in whole_list:
        average = (row[1] + row[2] + row[3]) / 3
        database.append([average])
    # print(database)
    # 月份
    for row in range(len(whole_list)):
        # print(whole_list[row][4].strip())
        if whole_list[row][4].strip() == "36 months":
            temp = 36
        elif whole_list[row][4].strip() == "60 months":
            temp = 60
        else:
            print(row[4].strip())
        database[row].append(temp)
    # print(database)
    # 利率
    for row in range(len(whole_list)):
        database[row].append(whole_list[row][5])
    # print(database)
    # 月供
    for row in range(len(whole_list)):
        database[row].append(whole_list[row][6])
    # print(database)
    # 信用
    for row in range(len(whole_list)):
        database[row].append((ord(whole_list[row][7]) - 64) * 10)
    # print(database)
    # sub_grade
    for row in range(len(whole_list)):
        back = ord(whole_list[row][8][1]) - 48
        front = ord(whole_list[row][8][0]) - 64
        # print(back * 2)
        database[row].append(front * 10 + back * 2 - 1)
    # print(database)
    # 岗位
    temp = 0
    for row in range(len(whole_list)):
        dict = {}
        dict[whole_list[row][9]] = dict.get(whole_list[row][9], -1)
        if dict[whole_list[row][9]] == -1:
            dict[whole_list[row][9]] = temp
            temp += 2
        else:
            print("2")
            back
        database[row].append(dict.get(whole_list[row][9]))
    #print(database)
    #就业时间长短
    test = []
    for row in range(len(whole_list)):
        dict = {'6 years', '2 years', '8 years', '4 years', '5 years', '< 1 year', '7 years', '9 years', '10+ years', '3 years', '1 year'}
        print(whole_list[row][10])
        test.append(whole_list[row][10])
        #database[row].append(dict.get(whole_list[row][9]))
    set = set(test)
    print(set)