import pandas
import numpy as np

pd = pandas.read_csv(r"./data/test1.csv")
array = np.array(pd)
#print(pd._ixs(6,2)) #获取一列
print(array)
