import pandas
import numpy as np

test = {}
temp = {}
for i in range(6):
    test[i] = temp
for i in range(6):
    for j in range(10,16):
        test[i][j]=99

print(test)