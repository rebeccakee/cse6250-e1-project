

"""
Read images and corresponding labels.
"""

import numpy as np
import os
import sys
import json
from torch.utils.data import Dataset
sys.path.append('loaddata')


class MyDataSet(Dataset):
    def __init__(self, 
            patient_list, 
            patient_time_record_dict, 
            patient_label_dict,
            patient_master_dict,
            phase='train', 
            split_num=5,           
            arg_list=None               
            ):

        self.patient_list = patient_list
        self.patient_time_record_dict = patient_time_record_dict
        self.patient_label_dict = patient_label_dict
        self.patient_master_dict = patient_master_dict
        self.phase = phase
        self.split_num = split_num
        self.split_nor = arg_list.split_nor
        self.split_nn = arg_list.split_nn
        self.args = arg_list
        self.time_range = arg_list.time_range
        self.last_time = arg_list.last_time
        self.length = 24
        self.n_code = 8


    def get_visit_info_wi_normal_range_func(self, time_record_data_dict):
        time_data_list = sorted(time_record_data_dict.keys(), key=lambda s:float(s))
        max_time = float(time_data_list[-1])

        visit_data_list = []
        value_data_list = []
        mask_data_list = []
        time_data_list = []
        temp_n_code = self.n_code
        trend_data_list = []
        previous_value = [[[],[]] for _ in range(temp_n_code)]
        change_th = 0.02
        start_time = - self.args.avg_time * 2
        end_time = -1
        feature_last_value = dict()
        init_data = []
        self.for_method_main(time_record_data_dict, time_data_list, max_time, visit_data_list, value_data_list, mask_data_list, temp_n_code, trend_data_list, previous_value, change_th, start_time, feature_last_value, init_data)

        num_len = self.length 

        if len(visit_data_list) <= num_len:
            visit = np.zeros(temp_n_code, dtype=np.int64)
            trend = np.zeros(temp_n_code, dtype=np.int64)
            value = np.zeros((2, temp_n_code), dtype=np.int64)
            while len(visit_data_list) < num_len:
                visit_data_list.append(visit)
                value_data_list.append(value)
                mask_data_list.append(visit)
                time_data_list.append(0)
                trend_data_list.append(trend)
                init_data.append([])
        else:
            visit_data_list = visit_data_list[- self.length:]
            value_data_list = value_data_list[- self.length:]
            mask_data_list = mask_data_list[- self.length:]
            time_data_list = time_data_list[- self.length:]
            trend_data_list = trend_data_list[- self.length:]
            init_data = init_data[- self.length :]


        return np.array(visit_data_list), np.array(value_data_list), np.array(mask_data_list, dtype=np.float32), np.array(time_data_list, dtype=np.float32), np.array(trend_data_list), init_data

    def for_method_main(self, time_record_data_dict, time_list, max_time, visit_list, value_list, mask_list, n_code, trend_list, previous_value, change_th, start_time, feature_last_value, init_data):
        for time in time_list :
            if float(time) <= - self.time_range:
                continue
            if float(time) - max_time >= self.last_time * 60:
                continue
            time = str(time)
            records = time_record_data_dict[time].items()
            feature_index = [int(r[0]) for r in records]
            feature_value = [float(r[1]) for r in records]
            feature_index = np.array(feature_index)
            feature_value = np.array(feature_value)
            feature = feature_index * self.split_nn
            trend = np.zeros(n_code, dtype=np.int64)
            self.for_method_func1(previous_value, change_th, start_time, time, feature_index, feature_value, trend)

            visit = np.zeros(n_code, dtype=np.int64)
            mask = np.zeros(n_code, dtype=np.int64)
            self.for_method_func2(feature_last_value, feature_index, feature_value, feature, visit, mask)
                    

            value = np.zeros((2, n_code ), dtype=np.int64)
            value[0][: len(feature_index)] = feature_index + 1
            value[1][: len(feature_index)] = (feature_value * self.args.n_split).astype(np.int64)
            value_list.append(value)
            visit_list.append(visit)
            mask_list.append(mask)
            time_list.append(float(time))
            trend_list.append(trend)
            init_data.append(dict(records))

    def for_method_func2(self, feature_last_value, feature_index, feature_value, feature, visit, mask):
        i_v = 0
        for feat, idx, val in zip(feature, feature_index,  feature_value):
            mask[i_v] = 1
            visit[i_v] = int(feat + 1)
            normal_range = [0.4, 0.6]
            range_value = 0
            if val < normal_range[0]:
                if normal_range[0]  > 0.1:
                    if val > normal_range[0]/2:
                        range_value += 1
            elif val > normal_range[1]:
                range_value += 3
                if normal_range[1]  < 0.9:
                    if 1 - val <  (1 - normal_range[1]) / 2:
                        range_value += 1
            else:
                range_value += 2

            delta_value = 1
            if self.args.use_trend:
                if idx in feature_last_value:
                    last_value = feature_last_value[idx]
                    delta = 0.3
                    if val - last_value < - delta:
                        delta_value = 0

                        feature_last_value[idx] = val
                    elif val - last_value > delta:
                        delta_value = 2

                        feature_last_value[idx] = val
                else:
                    feature_last_value[idx] = val

            visit[i_v] += range_value * 3 + delta_value
            i_v += 1

    def for_method_func1(self, previous_value, change_th, start_time, time, feature_index, feature_value, trend):
        i_v = 0
        for idx, val in zip(feature_index, feature_value):
            ptimes = previous_value[idx][0]
            lip = 0
            for ip, pt in enumerate(ptimes):
                if pt >= float(time) + start_time:
                    lip = ip
                    break
            avg_val = None
            if len(previous_value[idx][0]) == 1:
                avg_val = previous_value[idx][1][-1]
            previous_value[idx] = [
                        previous_value[idx][0][lip:],
                        previous_value[idx][1][lip:]]
            if len(previous_value[idx][0]):
                avg_val = np.mean(previous_value[idx][1])
            if avg_val is not None:
                if val < avg_val - change_th:
                    delta = 0
                elif val > avg_val + change_th:
                    delta = 1
                else:
                    delta = 2
                trend[i_v] = idx * 3 + delta + 1
            previous_value[idx][0].append(float(time))
            previous_value[idx][1].append(float(val))
            i_v += 1


    def __getitem__(self, index):
        patient = self.patient_list[index]
        if self.args.use_visit:
            visit_list, value_list, mask_list, time_list, trend_list,  init_data= self.get_visit_info_wi_normal_range_func(self.patient_time_record_dict[patient])

            if os.path.exists(self.args.master_file):
                master = self.patient_master_dict[patient]
                master = [int(m) for m in master]
                master = np.float32(master)
            else:
                master = 0
            if self.args.final == 1:
                label = np.float32(0)
            else:
                label = np.float32(self.patient_label_dict[patient])

            if self.args.compute_weight and self.phase != 'train' and label>0:
                with open(os.path.join(self.args.result_dir, 'cr', patient + '.init.json'), 'w') as f:
                    f.write(json.dumps(init_data, indent=4))
            if visit_list.max() > 121:
                pass
            if self.phase == 'test':
                return visit_list, value_list, mask_list, master, label, time_list, trend_list, patient
            else:
                return visit_list, value_list, mask_list, master, label, time_list, trend_list, patient




    def __len__(self):
        return len(self.patient_list) 
