# coding=utf8

import os
import sys
import json
import time
import pandas as pd
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args

def time_to_min(t):
    t = t.replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    t = t / 60
    return int(t)

def gen_json_data():
    vital_file = args.vital_file
    patient_time_record_dict = dict()
    print('Reading data...')
    feature_index_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_index_dict.json'))
    index_feature_list = py_op.myreadjson(os.path.join(args.result_dir, 'index_feature_list.json'))
    feature_value_order_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_value_order_dict.json'))
    feature_value_order_dict = { str(feature_index_dict[k]):v for k,v in feature_value_order_dict.items()  if 'time' not in k}
    patient_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_time_dict.json'))
    print('End reading')
    for i_line, line in enumerate(open(vital_file)):
        if i_line and i_line % 10000 == 0:
            print("Processing... At line", i_line) 
        if i_line:
            data = line.strip().split(',')
            patient, time = data[:2]
            time = time_to_min(time)
            time = int(float(time))
            if patient not in patient_time_record_dict:
                patient_time_record_dict[patient] = dict()

            data = data[2:]
            vs = dict()
            for idx, val in enumerate(data):
                if len(val) == 0:
                    continue
                value_order = feature_value_order_dict[str(idx)]
                vs[idx] = float('{:3.3f}'.format(value_order[val]))
            patient_time_record_dict[patient][time - patient_time_dict[patient] - 1] = vs

    with open(os.path.join(args.result_dir, 'patient_time_record_dict.json'), 'w') as f:
        f.write(json.dumps(patient_time_record_dict))

def main():
    gen_json_data()
    print("gen_vital_feature: Done!")


if __name__ == '__main__':
    main()
