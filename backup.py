from util import *

lst = get_sno_test_set()

with open('inference.sh', 'w') as f:
    for no in lst:
        f.write(f'python inference2.py --ckpt_dir ckpts/{no}/epoch=499.ckpt --station_id {no}\n')