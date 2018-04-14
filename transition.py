import xlrd
import csv

def xlsx_to_csv():
    workbook = xlrd.open_workbook('./data/draw.xlsx')
    table = workbook.sheet_by_index(0)
    with open('./data/draw.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)

xlsx_to_csv()