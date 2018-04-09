import csv
import pandas as pd
data_path = r"./data/train.csv"
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

df = pd.read_csv(data_path)[:0]
namelist = list(df)
print(df)
print(namelist)