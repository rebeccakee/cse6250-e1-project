import os
from sklearn import metrics
import numpy as np
import torch
import loaddata
from tools import parse
from loaddata import data_function

arg_list = parse.args

def load_models_func(data_dict, model_fileData):
    all_dict = torch.load(model_fileData)
    data_dict['epoch'] = all_dict['epoch']
    data_dict['best_metric'] = all_dict['best_metric']
    data_dict['model'].load_state_dict(all_dict['state_dict'])

def save_models_func(data_dict, name_str='best.ckpt', folder_path='../data/models/'):
    temp_arg_list = data_dict['args']
    name_str = '{:s}-{:s}-nl-{:d}-snm-{:d}-snr-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-{:s}'.format(
            temp_arg_list.task, temp_arg_list.model,temp_arg_list.num_layers,
            temp_arg_list.split_num, temp_arg_list.split_nor, temp_arg_list.use_trend, 
            temp_arg_list.use_cat, temp_arg_list.last_time, temp_arg_list.embed_size, name_str)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    model_data = data_dict['model']
    state_data_dict = model_data.state_dict()
    for key_item in state_data_dict.keys():
        state_data_dict[key_item] = state_data_dict[key_item].cpu()
    all_data_dict = {
            'epoch': data_dict['epoch'],
            'args': data_dict['args'],
            'best_metric': data_dict['best_metric'],
            'state_dict': state_data_dict 
            }
    torch.save(all_data_dict, os.path.join(folder_path, name_str))

def compute_metric_func(output_list, label_list, time_data, loss_output_list,metric_data_dict, phase_type='train'):
    if phase_type != 'test':
        preds_data = output_list.data.cpu().numpy()
        label_list = label_list.data.cpu().numpy()
    else:
        preds_data = np.array(output_list)
    preds_data = preds_data.reshape(-1)
    label_list = label_list.reshape(-1)
    preds_data = preds_data[label_list>-0.5]
    label_data = label_list[label_list>-0.5]
    pred_data = preds_data > 0
    sum_t_p = (pred_data + label_data == 2).sum()
    sum_t_n = (pred_data + label_data == 0).sum()
    sum_f_p = (pred_data - label_data == 1).sum()
    sum_f_n = (pred_data - label_data ==-1).sum()
    sum_f_p = (pred_data - label_data == 1).sum()
    metric_data_dict['tp'] = metric_data_dict.get('tp', 0.0) + sum_t_p
    metric_data_dict['tn'] = metric_data_dict.get('tn', 0.0) + sum_t_n
    metric_data_dict['fp'] = metric_data_dict.get('fp', 0.0) + sum_f_p
    metric_data_dict['fn'] = metric_data_dict.get('fn', 0.0) + sum_f_n
    temp_loss_list = []
    for item_x in loss_output_list:
        if item_x != 0:
            temp_loss_list.append(item_x.data.cpu().numpy())
        else:
            temp_loss_list.append(item_x)
    metric_data_dict['loss'] = metric_data_dict.get('loss', []) +  [temp_loss_list]
    if phase_type != 'train':
        metric_data_dict['preds'] = metric_data_dict.get('preds', []) + list(preds_data)
        metric_data_dict['labels'] = metric_data_dict.get('labels', []) + list(label_data)
        


def print_metric_func(first_line_str, metric_data_dict, phase_type='train'):
    print(first_line_str)
    try:
        loss_arr = np.array(metric_data_dict['loss']).mean(0)
    except:
        loss_arr = np.array([0, 0, 0])
    t_p = metric_data_dict['tp']
    t_n = metric_data_dict['tn']
    f_p = metric_data_dict['fp']
    f_n = metric_data_dict['fn']
    accuracy_data = 1.0 * (t_p + t_n) / (t_p + t_n + f_p + f_n)
    recall_data = 1.0 * t_p / (t_p + f_n + 10e-20)
    precision_data = 1.0 * t_p / (t_p + f_p + 10e-20)
    f1score_data = 2.0 * recall_data * precision_data / (recall_data + precision_data + 10e-20)
    loss_arr = loss_arr.reshape(-1)
    print('loss: {:3.4f}\t pos loss: {:3.4f}\t negloss: {:3.4f}'.format(loss_arr[0], loss_arr[1], loss_arr[2]))
    print('accuracy: {:3.4f}\t f1score: {:3.4f}\t recall: {:3.4f}\t precision: {:3.4f}'.format(accuracy_data, f1score_data, recall_data, precision_data))
    print('\n')
    if phase_type == 'train':
        return f1score_data
    else:
        fpr, tpr, thr = metrics.roc_curve(metric_data_dict['labels'], metric_data_dict['preds'])
        return metrics.auc(fpr, tpr)
        
