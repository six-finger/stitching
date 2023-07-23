import csv

def read_csv_file(file_path):
    rows = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            rows.append(row)
    return rows

def get_specific_row(rows, row_index):
    return rows[row_index]

def get_specific_column(rows, col_index):
    return [row[col_index] for row in rows]

def get_specific_img(rows, img_name):
    '''
    return: 0,img_name; 1-3,gps; 4-6,pose.
    '''
    for row in rows:
        if img_name==row[0]:
            return row

