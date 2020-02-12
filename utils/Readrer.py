import csv

reader = []


def read_file(path):
    global reader
    table = []
    row_num = 0
    with open(path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            table.append(row)
            row_num += 1
    return table, row_num
