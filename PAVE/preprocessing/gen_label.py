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

def gen_patient_label_dict():
    patient_label_dict = dict()
    label_file = args.label_file
    for i_line,line in enumerate(open(label_file)):
        if i_line != 0:
            data = line.strip().split(',')
            patient = str(int(float(data[0])))
            # patient = data[0]
            label  = data[-1]
            patient_label_dict[patient] = int(float(label))
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_label_dict.json'), patient_label_dict)

    print('There are {:d} case samples.'.format(sum(patient_label_dict.values())))
    print('There are {:d} control samples.'.format(len(patient_label_dict) - sum(patient_label_dict.values())))

# def split_data():
#     patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
#     # patients = patient_label_dict.keys()
#     # patients = sorted(patients)
#     patients = py_op.myreadjson(os.path.join(args.result_dir, 'patient_list.json'))
#     n = int(len(patients) * 0.8)
#     patient_train = patients[:n]
#     patient_valid = patients[n:]
#     py_op.mywritejson(os.path.join(args.result_dir, 'train.json'), patient_train)
#     py_op.mywritejson(os.path.join(args.result_dir, 'valid.json'), patient_valid)
#     print(sum([patient_label_dict[k] for k in patient_train])) 
#     print(sum([patient_label_dict[k] for k in patient_valid])) 
#     print(len([patient_label_dict[k] for k in patient_train])) 

def main():
    gen_patient_label_dict()
    print("gen_label: Done!")
    # split_data()


if __name__ == '__main__':
    main()
