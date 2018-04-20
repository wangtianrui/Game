import csv
import pickle
import numpy as np
import pandas as pd

data_path = "./data/test.csv"
maker_dict = "./data/makerDict.txt"
log_paht = "./400000/"


def pickle_load(path):
    """
    读取字典
    :return:
    """
    # f = open("./data/maker.txt", "rb")
    f = open(path, "rb")
    maker_dicts = pickle.load(f)
    return maker_dicts


def readData(datapath):
    whole_ist = pd.read_csv(datapath, encoding='ISO-8859-1')
    whole_ist = whole_ist.fillna(0)
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
    return kind_64_dict, kind_index, namelist


