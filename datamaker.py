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
    # print(kind_64_dict)
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
    # print(database)
    # 就业时间长短
    dict = {'6 years': 6, '2 years': 2, '8 years': 8, '4 years': 4, '5 years': 5, '< 1 year': 0, '7 years': 7,
            ' ': 100, '9 years': 9, '10+ years': 10, '3 years': 3, '1 year': 1}
    for row in range(len(whole_list)):
        temp = dict[whole_list[row][10]]
        database[row].append(temp)
    # print(database)
    #    print(whole_list[row][10])
    #     test.append(whole_list[row][10])
    #     if whole_list[row][10] == ' ':
    #         count += 1
    #         # database[row].append(dict.get(whole_list[row][9]))
    # set = set(test)
    # print(set)
    # print(count)
    # 房屋所有情况
    dict = {"MORTGAGE": 0, "RENT": 50, "OWN": 100, "OTHER": 150}
    for row in range(len(whole_list)):
        temp = dict[whole_list[row][11]]
        database[row].append(temp)
    # 年收入
    for row in range(len(whole_list)):
        temp = whole_list[row][12]
        database[row].append(temp)
    # test = []
    # 信用是否核实
    dict = {'Source Verified': 0, 'Verified': 50, 'Not Verified': 100}
    for row in range(len(whole_list)):
        temp = whole_list[row][13]
        database[row].append(dict[temp])
    # 申请日期
    test = []
    dict = {'Sep': 9, 'Mar': 3, 'Dec': 12, 'Feb': 2, 'May': 5, 'Apr': 4, 'Jun': 6, 'Nov': 11, 'Aug': 8, 'Jan': 1,
            'Jul': 7, 'Oct': 10}
    for row in range(len(whole_list)):
        temp = whole_list[row][14]
        yue = dict[temp[:3]]
        ri = int(temp[4:])
        database[row].append(int(str(yue) + str(ri)))
    # 贷款现状
    dict = {'Does not meet the credit policy. Status:Charged Off': 0, 'Issued': 1, 'Current': 2, 'Charged Off': 3,
            'In Grace Period': 4, 'Late (16-30 days)': 5, 'Default': 10,
            'Does not meet the credit policy. Status:Fully Paid': 8,
            'Fully Paid': 6, 'Late (31-120 days)': 7}
    for row in range(len(whole_list)):
        temp = whole_list[row][15]
        database[row].append(dict[whole_list[row][15]])

        # database[row].append(int(str(yue) + str(ri)))
    # purpose
    test = []
    dict = {'vacation': 0, 'credit_card': 1, 'house': 2, 'medical': 3, 'wedding': 4, 'major_purchase': 5,
            'renewable_energy': 6,
            'educational': 7, 'car': 8, 'debt_consolidation': 9, 'moving': 10, 'small_business': 11,
            'home_improvement': 12, 'other': 13}
    for row in range(len(whole_list)):
        temp = whole_list[row][18]
        database[row].append(dict[temp])

    # title
    test = []
    dict = {}
    temp = 0
    for row in range(len(whole_list)):
        dict[whole_list[row][19]] = dict.get(whole_list[row][19], -1)
        if dict[whole_list[row][19]] == -1:
            dict[whole_list[row][19]] = temp
            temp += 1
        database[row].append(dict[whole_list[row][19]])


    #print(len(set(test)))
    print(database)
