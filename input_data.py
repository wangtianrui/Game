import csv

data_path = r"./data/train.csv"
csv_reader = csv.reader(open(data_path, encoding='utf-8'))
list = list(csv_reader)
print(list[0])
print(len(list[0]))