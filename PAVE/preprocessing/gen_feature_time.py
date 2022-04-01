# coding=utf8

import os
import sys
import time
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args

def time_to_min(t):
    t = t.replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    t = t / 60
    return int(t)

def gen_patient_time_dict():
    vital_file = args.vital_file
    patient_time_dict = dict()
    for i_line,line in enumerate(open(vital_file)):
        if i_line % 10000 == 0:
            print("Processing... At line", i_line) 
        if i_line:
            patient, time = line.strip().split(',')[:2]
            time = time_to_min(time)
            patient_time_dict[patient] = max(patient_time_dict.get(patient, 0), time)
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_time_dict.json'), patient_time_dict)

def gen_feature_index():
    vital_file = args.vital_file
    for i_line,line in enumerate(open(vital_file)):
        line = line.replace('"', '')
        index_feature_list = line.strip().split(',')[2:]
        break
    feature_index_dict = { f:i for i,f in enumerate(index_feature_list) }

    py_op.mywritejson(os.path.join(args.result_dir, 'feature_index_dict.json'), feature_index_dict)
    py_op.mywritejson(os.path.join(args.result_dir, 'index_feature_list.json'), index_feature_list)


def main():
    gen_patient_time_dict()
    gen_feature_index()
    print("gen_feature_time: Done!")


if __name__ == '__main__':
    main()
