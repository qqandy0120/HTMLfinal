import os
import pandas as pd
import numpy as ap
from datetime import datetime
PRED_DATE = [f'2023102{no}' for no in range(1,5)] + [f'202312{no}' for no in range(11,18)]

def sort_result(line):
    # 20231021_500101001_00:00,1.0
    id, sbi = line.split(',')
    date, station_id, time = id.split('_')
    return (date, station_id, time)

file_lst = os.listdir('./outputs')

result_lst = []

for file in file_lst:
    date = file.split('_')[1]
    if date not in ['1021', '1211']:
        continue
    else:
        with open(os.path.join('outputs', file), 'r') as r_f:
            lines = r_f.readlines()
            for line in lines[1:]:
                result_lst.append(line)


result_lst = sorted(result_lst, key=sort_result)

with open('final_result_s2.csv', 'w') as w_f:
    w_f.write('id, sbi\n')
    for line in result_lst:
        id, sbi = line.replace('\n', '').split(',')
        date = id.split('_')[0]
        if date in PRED_DATE:
            w_f.write(f'{id},{max(0, float(sbi))}\n')
        