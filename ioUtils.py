import csv
import pickle
import numpy as np
import pandas as pd


def pickle_save(maker_dicts,path):
    """
    保存maker字典
    :return:
    """
    #f = open("./data/maker2.txt", "wb")
    f = open(path, "wb")
    pickle.dump(maker_dicts, f)
    f.close()


def pickle_load(path):
    """
    读取字典
    :return:
    """
    #f = open("./data/maker.txt", "rb")
    f = open(path, "rb")
    maker_dicts = pickle.load(f)
    return maker_dicts


def readData(datapath):
    whole_ist = pd.read_csv(datapath, encoding='ISO-8859-1')
    whole_ist = whole_ist.fillna(0)
    #print(whole_ist)
    return whole_ist


def getNameDict(wholeList):
    """
    获取第一行信息
    :param wholeList:
    :return:一个空的64类对应的字典，一个64名字的索引字典，一个64类名字的list
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

