import os
import pandas as pd
import numpy as ap
from datetime import datetime

def sort_file_order(file_list):
    """ Sorts filenames by descending date and then ascending id for the same date. """
    # Parsing the date and id from the filename
    def extract_date_id(filename):
        parts = filename.split('_')
        date = datetime.strptime(parts[1], "%m%d")
        id = int(parts[2].split('.')[0])
        return date, id

    # Sorting by descending date and then ascending id
    return sorted(file_list, key=lambda x: (extract_date_id(x)[0], -extract_date_id(x)[1]), reverse=True)

file_lst = sort_file_order(os.listdir('./outputs'))
ids, sbis = [], []
with open('final_result_2.csv', 'w') as w_f:
    w_f.write("id,sbi\n")
    for file in file_lst:
        with open(f'./outputs/{file}', 'r') as r_f:
            for i, line in enumerate(r_f.readlines()):
                if i != 0:
                    line = f'{line.split(",")[0]},{max(float(line.split(",")[1]), 0)}\n'
                    w_f.write(line)
print('done!')