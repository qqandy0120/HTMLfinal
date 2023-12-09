import typing 
import json
from typing import List, Dict
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def get_sno_test_set() -> List[str]:
    with open("html.2023.final.data/sno_test_set.txt", "r") as f:
        lines = f.readlines()
    return [str(no.replace('\n', '')) for no in lines]

def get_date_list() -> List[str]:
    return sorted(os.listdir(os.path.join("html.2023.final.data", "release")))

def add_one_minute(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M')
    time_obj += timedelta(minutes=1)
    return time_obj.strftime('%H:%M')

def delta_time(time_str1, time_str2):

    time_obj1 = datetime.strptime(time_str1, '%H:%M')
    time_obj2 = datetime.strptime(time_str2, '%H:%M')

    if time_obj2 < time_obj1:
        time_difference = (time_obj2 - time_obj1).seconds
    else:
        time_difference = (time_obj2 - time_obj1).seconds

    minutes = time_difference // 60

    return minutes

def cyclical_encode_time(hour, minute):
    sin_hour = np.sin(2 * np.pi * hour/23.0)
    cos_hour = np.cos(2 * np.pi * hour/23.0)
    sin_minute = np.sin(2 * np.pi * minute/59.0)
    cos_minute = np.cos(2 * np.pi * minute/59.0)
    return round(sin_hour, 5), round(cos_hour, 5), round(sin_minute, 5), round(cos_minute, 5)

def generate_time_period_list(interval_minutes):
    start_time = datetime.strptime("00:00", "%H:%M")
    interval = timedelta(minutes=interval_minutes)
    time_strings = []

    while start_time < datetime.strptime("23:59", "%H:%M"):
        time_strings.append(start_time.strftime("%H:%M"))
        start_time += interval

    return time_strings

def get_stat_raw_data(station_id, split, time_step):
    all_raw_data = []
    if split == 'train':
        date_list = get_date_list()[7:]
    if split == 'valid':
        date_list = get_date_list()[:7]
    if split == 'test_1021':
        date_list = ['20231020']
        pred_date = ['20231021', '20231022', '20231023', '20231024']
    if split == 'test_1204':
        date_list = ['20231126']  # need to be update when 1203 data are released
        pred_date = [f'202312{str(no).zfill(2)}' for no in range(4, 11)]  # 1204 to 1210
    if split == 'test_1211':
        date_list = ['20231208']
        pred_date = [f'202312{str(no).zfill(2)}' for no in range(9, 18)]


    for date in date_list:
            day = str(datetime.strptime(date, '%Y%m%d').weekday())
            file_path = os.path.join("html.2023.final.data", "release", date, f'{station_id}.json')
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            for time, attr in data.items():
                hour, minute = [int(no) for no in time.split(':')]
                if time.endswith(('00','20','40')):
                    # fill with the nearest future data
                    not_found = False
                    if not attr:
                        future_time = time
                        while not data[future_time]:
                            future_time = add_one_minute(future_time)
                            if future_time == '00:00':
                                not_found = True
                        value = data[future_time]['sbi'] if not not_found else -1
                    else:
                        value = attr['sbi']
                
                    
                    sin_hour, cos_hour, sin_minute, cos_minute = cyclical_encode_time(hour,minute)

                    all_raw_data.append({'id': f'{date}_{station_id}_{time}',
                                        'Monday': 1 if day == '0' else 0,
                                        'Tuesday': 1 if day == '1' else 0,
                                        'Wednesday': 1 if day == '2' else 0,
                                        'Thursday': 1 if day == '3' else 0,
                                        'Friday': 1 if day == '4' else 0,
                                        'Saturday': 1 if day == '5' else 0,
                                        'Sunday': 1 if day == '6' else 0,
                                        's_hour': sin_hour,
                                        'c_hour': cos_hour,
                                        's_min': sin_minute,
                                        'c_min': cos_minute,
                                        'sbi': value,
                                        })
    
    if split not in ['train', 'valid']:
        all_raw_data = all_raw_data[-6:]

        for date in pred_date:
            for time in generate_time_period_list(interval_minutes=20):
                day = str(datetime.strptime(date, '%Y%m%d').weekday())
                hour, minute = [int(no) for no in time.split(':')]
                all_raw_data.append({'id': f'{date}_{station_id}_{time}',
                                        'Monday': 1 if day == '0' else 0,
                                        'Tuesday': 1 if day == '1' else 0,
                                        'Wednesday': 1 if day == '2' else 0,
                                        'Thursday': 1 if day == '3' else 0,
                                        'Friday': 1 if day == '4' else 0,
                                        'Saturday': 1 if day == '5' else 0,
                                        'Sunday': 1 if day == '6' else 0,
                                        's_hour': sin_hour,
                                        'c_hour': cos_hour,
                                        's_min': sin_minute,
                                        'c_min': cos_minute,
                                        })


    return pd.DataFrame(all_raw_data)


if __name__ == '__main__':
    # a = get_stat_raw_data(station_id='500101001', split='test_1021', time_step=6)
    # print(a)

    lst = generate_time_period_list(interval_minutes=20)
    print(lst)